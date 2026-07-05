use arrayvec::ArrayVec;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{GRID_SIZE, HexCoord, Rotation, StarterPlacement, Terrain, Tile, Wildlife};

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

/// Per-edge neighbor habitat facts for one candidate tile cell; see
/// [`HabitatAnalysis::tile_neighbor_context`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileNeighborContext {
    /// For each edge: the neighbor tile's facing terrain, its component id
    /// for that terrain, and that component's size.
    edges: [Option<(Terrain, u8, u8)>; 6],
}

/// Precomputed answer structure for repeated
/// [`HabitatAnalysis::largest_after_tile_with_context`] probes at one cell:
/// for each terrain, the distinct neighboring components facing the cell as
/// `(edge_mask, size)` pairs. See [`HabitatAnalysis::cell_habitat_query`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CellHabitatQuery {
    components: [ArrayVec<(u8, u8), 6>; 5],
    largest: [u8; 5],
}

impl CellHabitatQuery {
    /// Largest habitat corridor for `terrain` after placing a tile at this
    /// cell that shows `terrain` on the edges of `edge_mask`. Equivalent to
    /// [`HabitatAnalysis::largest_after_tile_with_context`] for a tile that
    /// contains `terrain`: every distinct neighboring component of `terrain`
    /// reachable through one of the masked edges contributes its size once.
    #[inline]
    pub fn largest_after_terrain_edges(&self, terrain: Terrain, edge_mask: u8) -> u8 {
        let mut size = 1u8;
        for &(mask, component_size) in &self.components[terrain as usize] {
            if mask & edge_mask != 0 {
                size += component_size;
            }
        }
        self.largest[terrain as usize].max(size)
    }
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

    /// Captures, for one candidate cell, the habitat facts of the six
    /// neighboring tiles that [`Self::largest_after_tile`] reads: the terrain
    /// each neighbor shows on the facing edge plus that terrain's component id
    /// and size. The context only depends on the neighboring cells, so it can
    /// be computed once per candidate coordinate and reused across every
    /// tile, rotation, and terrain probed there.
    pub fn tile_neighbor_context(&self, board: &Board, coord: HexCoord) -> TileNeighborContext {
        let mut edges = [None; 6];
        for (edge, slot) in edges.iter_mut().enumerate() {
            let Some(neighbor_index) = coord.neighbor(edge).to_index() else {
                continue;
            };
            let Some(neighbor_tile) = board.cells[neighbor_index] else {
                continue;
            };
            let facing = neighbor_tile
                .tile
                .terrain_on_edge(neighbor_tile.rotation, (edge + 3) % 6);
            let component = self.component_ids[facing as usize][neighbor_index];
            *slot = Some((
                facing,
                component,
                self.component_sizes[facing as usize][usize::from(component)],
            ));
        }
        TileNeighborContext { edges }
    }

    /// Equivalent of [`Self::largest_after_tile`] evaluated against a
    /// prebuilt [`TileNeighborContext`] for the same board and coordinate.
    pub fn largest_after_tile_with_context(
        &self,
        context: &TileNeighborContext,
        tile: Tile,
        rotation: Rotation,
        terrain: Terrain,
    ) -> u8 {
        if !tile.contains_terrain(terrain) {
            return self.largest(terrain);
        }

        let mut connected_components = [0u8; 6];
        let mut connected_count = 0usize;
        let mut component_size = 1u8;
        for edge in 0..6 {
            if tile.terrain_on_edge(rotation, edge) != terrain {
                continue;
            }
            let Some((facing, component, size)) = context.edges[edge] else {
                continue;
            };
            if facing != terrain
                || component == 0
                || connected_components[..connected_count].contains(&component)
            {
                continue;
            }
            connected_components[connected_count] = component;
            connected_count += 1;
            component_size += size;
        }
        self.largest(terrain).max(component_size)
    }

    /// Builds a [`CellHabitatQuery`] for one candidate cell: the distinct
    /// neighboring habitat components per terrain, each with the 6-bit mask
    /// of edges on which it faces the cell and its size. Combined with
    /// [`Tile::terrain_edge_mask`] this answers
    /// [`Self::largest_after_tile_with_context`] with a couple of mask tests
    /// instead of a per-call edge scan; the produced sizes are bit-identical.
    pub fn cell_habitat_query(&self, context: &TileNeighborContext) -> CellHabitatQuery {
        let mut components: [ArrayVec<(u8, u8, u8), 6>; 5] = Default::default();
        for (edge, slot) in context.edges.iter().enumerate() {
            let Some((facing, component, size)) = *slot else {
                continue;
            };
            if component == 0 {
                continue;
            }
            let bucket = &mut components[facing as usize];
            if let Some(entry) = bucket.iter_mut().find(|(id, ..)| *id == component) {
                entry.1 |= 1 << edge;
            } else {
                bucket.push((component, 1 << edge, size));
            }
        }
        CellHabitatQuery {
            components: components.map(|bucket| {
                bucket
                    .into_iter()
                    .map(|(_, edge_mask, size)| (edge_mask, size))
                    .collect()
            }),
            largest: self.largest,
        }
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
            rotation,
            wildlife: None,
        });
        self.placed_indices.push(index as u16);
        Ok(BoardDelta::TilePlaced { index })
    }

    /// [`Self::place_tile`] for coordinates already known to touch the
    /// existing environment (the frontier of this board): skips only the
    /// neighbor attachment scan, which by construction succeeds for frontier
    /// cells. All other checks and the resulting board state are identical.
    pub(crate) fn place_tile_attached(
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
        debug_assert!(
            self.placed_indices.is_empty()
                || coord.neighbors().into_iter().any(|neighbor| {
                    neighbor
                        .to_index()
                        .is_some_and(|neighbor_index| self.cells[neighbor_index].is_some())
                }),
            "place_tile_attached requires a frontier coordinate"
        );
        self.cells[index] = Some(PlacedTile {
            tile,
            rotation,
            wildlife: None,
        });
        self.placed_indices.push(index as u16);
        Ok(BoardDelta::TilePlaced { index })
    }

    /// Rewrites the stored rotation of the tile at `index`. Used by the legal
    /// action visitor to probe every rotation of a candidate tile without
    /// re-running the placement checks per rotation; the board state is
    /// identical to removing the tile and re-placing it with `rotation`.
    pub(crate) fn set_placed_rotation(&mut self, index: usize, rotation: Rotation) {
        self.cells[index]
            .as_mut()
            .expect("rotation updates target an occupied cell")
            .rotation = rotation;
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
            if self.cells[usize::from(index)].is_none() {
                return Err("placed index points to an empty cell");
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
            rotation,
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
            rotation,
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
    fn terrain_edge_masks_match_terrain_on_edge() {
        for tile in STANDARD_TILES.iter() {
            for rotation in Rotation::ALL {
                let (mask_a, mask_b) = tile.terrain_edge_masks(rotation);
                assert_eq!(mask_a | mask_b, 0b11_1111);
                assert_eq!(mask_a & mask_b, 0);
                for edge in 0..6 {
                    // Dual tiles never repeat a terrain in the catalog, so
                    // the edge terrain decides mask membership; keystone
                    // tiles show terrain_a everywhere.
                    let terrain = tile.terrain_on_edge(rotation, edge);
                    assert_eq!(
                        mask_a & (1 << edge) != 0,
                        terrain == tile.terrain_a,
                        "tile {:?} rotation {rotation:?} edge {edge}",
                        tile.id
                    );
                }
            }
        }
    }

    #[test]
    fn cell_habitat_query_matches_context_probe_on_grown_boards() {
        for (cluster_index, starter) in STARTER_CLUSTERS.iter().enumerate() {
            let mut board = Board::from_starter(starter);
            // Grow the board deterministically so queries see multi-tile
            // components approached from several edges.
            for step in 0..12usize {
                let frontier = board.frontier();
                let coord = frontier[(step * 7 + cluster_index) % frontier.len()];
                let tile = STANDARD_TILES[(step * 11 + cluster_index * 3) % STANDARD_TILES.len()];
                let rotation = Rotation::ALL[step % 6];
                board.place_tile(coord, tile, rotation).unwrap();
            }
            let analysis = board.habitat_analysis();
            for coord in board.frontier() {
                let context = analysis.tile_neighbor_context(&board, coord);
                let query = analysis.cell_habitat_query(&context);
                for tile in STANDARD_TILES.iter().take(30) {
                    for rotation in Rotation::ALL {
                        let (mask_a, mask_b) = tile.terrain_edge_masks(rotation);
                        assert_eq!(
                            query.largest_after_terrain_edges(tile.terrain_a, mask_a),
                            analysis.largest_after_tile_with_context(
                                &context,
                                *tile,
                                rotation,
                                tile.terrain_a
                            ),
                        );
                        if let Some(terrain_b) = tile.terrain_b {
                            assert_eq!(
                                query.largest_after_terrain_edges(terrain_b, mask_b),
                                analysis.largest_after_tile_with_context(
                                    &context, *tile, rotation, terrain_b
                                ),
                            );
                        }
                    }
                }
            }
        }
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
