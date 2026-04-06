use arrayvec::ArrayVec;

use crate::hex::{HexCoord, HexGrid, ADJACENCY, GRID_SIZE};
use crate::types::{Cell, Terrain, TileData, Wildlife};

/// Disjoint-set (union-find) with path compression and union by rank.
/// Fixed-size, inline — no heap allocation.
#[derive(Clone)]
pub struct UnionFind {
    parent: [u16; GRID_SIZE],
    rank: [u8; GRID_SIZE],
    size: [u16; GRID_SIZE],
}

impl UnionFind {
    pub fn new() -> Self {
        let mut uf = UnionFind {
            parent: [0; GRID_SIZE],
            rank: [0; GRID_SIZE],
            size: [0; GRID_SIZE],
        };
        for i in 0..GRID_SIZE {
            uf.parent[i] = i as u16;
        }
        uf
    }

    /// Find root with path compression.
    #[inline]
    pub fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] as usize != x {
            // Path splitting (single-pass path compression)
            let p = self.parent[x] as usize;
            self.parent[x] = self.parent[p];
            x = p;
        }
        x
    }

    /// Union two elements. Returns the root of the merged set.
    #[inline]
    pub fn union(&mut self, a: usize, b: usize) -> usize {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return ra;
        }

        // Union by rank
        let (big, small) = if self.rank[ra] >= self.rank[rb] {
            (ra, rb)
        } else {
            (rb, ra)
        };
        self.parent[small] = big as u16;
        self.size[big] += self.size[small];
        if self.rank[big] == self.rank[small] {
            self.rank[big] += 1;
        }
        big
    }

    /// Mark a cell as active in this union-find (size = 1).
    #[inline]
    pub fn activate(&mut self, idx: usize) {
        self.parent[idx] = idx as u16;
        self.rank[idx] = 0;
        self.size[idx] = 1;
    }

    /// Get the size of the set containing idx.
    #[inline]
    pub fn set_size(&mut self, idx: usize) -> u16 {
        let root = self.find(idx);
        self.size[root]
    }

    /// Check if a cell is active (has been placed).
    #[inline]
    pub fn is_active(&self, idx: usize) -> bool {
        self.size[idx] > 0 || {
            // If size is 0 but it's a root pointing to itself with rank 0,
            // it hasn't been activated. Check via size of root.
            false
        }
    }
}

/// Record of an action for undo support.
#[derive(Debug, Clone)]
pub enum UndoAction {
    PlaceTile {
        idx: usize,
        /// Snapshot of union-find states affected (terrain indices that were unioned)
        habitat_merges: ArrayVec<HabitatMerge, 12>, // up to 6 merges × 2 terrains
        largest_group_before: [u16; 5],
    },
    PlaceWildlife {
        idx: usize,
        wildlife: Wildlife,
    },
}

#[derive(Debug, Clone, Copy)]
pub struct HabitatMerge {
    pub terrain: u8,
    pub parent_before: [u16; 2], // parent[a_root], parent[b_root] before union
    pub rank_before: [u8; 2],
    pub size_before: [u16; 2],
    pub roots: [usize; 2], // the two roots before union
}

/// For a dual-terrain tile at a given rotation, determine which terrain
/// faces a given edge direction.
///
/// A dual-terrain tile has terrain1 on 3 consecutive edges and terrain2 on the other 3.
/// At rotation R, terrain1 is on edges R, R+1, R+2 (mod 6) and terrain2 on R+3, R+4, R+5.
/// Single-terrain tiles return the same terrain for all edges.
#[inline]
pub fn terrain_on_edge(cell: Cell, rotation: u8, direction: usize) -> Option<Terrain> {
    if !cell.is_present() {
        return None;
    }
    let t1 = cell.primary_terrain();
    let t2 = cell.secondary_terrain();
    if t2.is_none() {
        // Single-terrain (keystone) — all edges are terrain1
        return t1;
    }
    // Dual-terrain: edges [rot, rot+1, rot+2] % 6 are terrain1, rest are terrain2
    let offset = ((direction as i8 - rotation as i8).rem_euclid(6)) as u8;
    if offset < 3 { t1 } else { t2 }
}

/// The game board for a single player.
#[derive(Clone)]
pub struct Board {
    pub grid: HexGrid,
    pub habitat_uf: [UnionFind; 5],
    pub largest_group: [u16; 5],
    /// Positions of each wildlife type (for fast scoring iteration).
    pub wildlife_positions: [ArrayVec<u16, 24>; 5],
    pub tile_count: u8,
    pub nature_tokens: u8,
    /// Rotation (0-5) of each placed tile.
    pub rotations: [u8; GRID_SIZE],
    /// Stack of placed tile indices for frontier computation.
    pub placed_tiles: ArrayVec<u16, 64>,
}

impl Board {
    pub fn new() -> Self {
        Board {
            grid: HexGrid::new(),
            habitat_uf: std::array::from_fn(|_| UnionFind::new()),
            largest_group: [0; 5],
            wildlife_positions: std::array::from_fn(|_| ArrayVec::new()),
            tile_count: 0,
            nature_tokens: 0,
            rotations: [0; GRID_SIZE],
            placed_tiles: ArrayVec::new(),
        }
    }

    /// Place a tile at the given coordinate with a rotation (0-5).
    /// Rotation determines which edges have which terrain for dual-terrain tiles.
    /// Returns an UndoAction for backtracking.
    pub fn place_tile(&mut self, coord: HexCoord, tile: TileData, rotation: u8) -> Option<UndoAction> {
        let idx = coord.to_index()?;
        let rotation = rotation % 6;

        // Don't place on occupied cell
        if self.grid.get(idx).is_present() {
            return None;
        }

        let cell = tile.to_cell();
        self.grid.set(idx, cell);
        self.rotations[idx] = rotation;
        self.tile_count += 1;
        self.placed_tiles.push(idx as u16);

        let largest_before = self.largest_group;
        let mut merges: ArrayVec<HabitatMerge, 12> = ArrayVec::new();

        // Process each terrain on this tile
        let terrains: ArrayVec<Terrain, 2> = {
            let mut t = ArrayVec::new();
            t.push(tile.terrain1);
            if let Some(t2) = tile.terrain2 {
                t.push(t2);
            }
            t
        };

        let adj = &*ADJACENCY;
        for &terrain in &terrains {
            let ti = terrain as usize;
            self.habitat_uf[ti].activate(idx);

            // Union with adjacent cells that share this terrain on the shared edge
            for dir in 0..6 {
                let nidx_val = adj.neighbors[idx][dir];
                if nidx_val == u16::MAX {
                    continue;
                }
                let nidx = nidx_val as usize;
                let ncell = self.grid.get(nidx);
                if !ncell.is_present() {
                    continue;
                }

                // Check if this tile has `terrain` on edge `dir`
                let my_terrain_on_edge = terrain_on_edge(cell, rotation, dir);
                if my_terrain_on_edge != Some(terrain) {
                    continue;
                }

                // Check if neighbor has `terrain` on the opposite edge (dir+3)%6
                let opposite_dir = (dir + 3) % 6;
                let neighbor_rot = self.rotations[nidx];
                let neighbor_terrain_on_edge = terrain_on_edge(ncell, neighbor_rot, opposite_dir);
                if neighbor_terrain_on_edge != Some(terrain) {
                    continue;
                }

                let root_a = self.habitat_uf[ti].find(idx);
                let root_b = self.habitat_uf[ti].find(nidx);
                if root_a != root_b {
                    let merge = HabitatMerge {
                        terrain: ti as u8,
                        parent_before: [
                            self.habitat_uf[ti].parent[root_a],
                            self.habitat_uf[ti].parent[root_b],
                        ],
                        rank_before: [
                            self.habitat_uf[ti].rank[root_a],
                            self.habitat_uf[ti].rank[root_b],
                        ],
                        size_before: [
                            self.habitat_uf[ti].size[root_a],
                            self.habitat_uf[ti].size[root_b],
                        ],
                        roots: [root_a, root_b],
                    };
                    merges.push(merge);
                    self.habitat_uf[ti].union(root_a, root_b);
                }
            }

            // Update largest group for this terrain
            let current_size = self.habitat_uf[ti].set_size(idx);
            if current_size > self.largest_group[ti] {
                self.largest_group[ti] = current_size;
            }
        }

        Some(UndoAction::PlaceTile {
            idx,
            habitat_merges: merges,
            largest_group_before: largest_before,
        })
    }

    /// Place a wildlife token at the given index.
    pub fn place_wildlife(&mut self, idx: usize, wildlife: Wildlife) -> Option<UndoAction> {
        let cell = self.grid.get(idx);
        if !cell.can_place_wildlife(wildlife) {
            return None;
        }

        self.grid.set(idx, cell.with_wildlife(wildlife));
        self.wildlife_positions[wildlife as usize].push(idx as u16);

        // Award nature token if keystone tile
        if cell.is_keystone() {
            self.nature_tokens += 1;
        }

        Some(UndoAction::PlaceWildlife { idx, wildlife })
    }

    /// Undo a previous action (for search backtracking).
    pub fn undo(&mut self, action: UndoAction) {
        match action {
            UndoAction::PlaceTile {
                idx,
                habitat_merges,
                largest_group_before,
            } => {
                // Restore largest group
                self.largest_group = largest_group_before;

                // Undo union-find merges in reverse order
                for merge in habitat_merges.iter().rev() {
                    let ti = merge.terrain as usize;
                    let [ra, rb] = merge.roots;
                    self.habitat_uf[ti].parent[ra] = merge.parent_before[0];
                    self.habitat_uf[ti].parent[rb] = merge.parent_before[1];
                    self.habitat_uf[ti].rank[ra] = merge.rank_before[0];
                    self.habitat_uf[ti].rank[rb] = merge.rank_before[1];
                    self.habitat_uf[ti].size[ra] = merge.size_before[0];
                    self.habitat_uf[ti].size[rb] = merge.size_before[1];
                }

                // Deactivate the cell in all relevant union-finds
                let cell = self.grid.get(idx);
                if let Some(t) = cell.primary_terrain() {
                    self.habitat_uf[t as usize].size[idx] = 0;
                }
                if let Some(t) = cell.secondary_terrain() {
                    self.habitat_uf[t as usize].size[idx] = 0;
                }

                // Remove tile
                self.grid.set(idx, Cell::EMPTY);
                self.rotations[idx] = 0;
                self.tile_count -= 1;
                self.placed_tiles.pop();
            }
            UndoAction::PlaceWildlife { idx, wildlife } => {
                let cell = self.grid.get(idx);

                // Remove nature token if keystone
                if cell.is_keystone() {
                    self.nature_tokens -= 1;
                }

                self.grid.set(idx, cell.without_wildlife());
                let positions = &mut self.wildlife_positions[wildlife as usize];
                if let Some(pos) = positions.iter().position(|&p| p == idx as u16) {
                    positions.swap_remove(pos);
                }
            }
        }
    }

    /// Get all frontier positions (empty cells adjacent to at least one placed tile).
    pub fn frontier(&self) -> ArrayVec<u16, 128> {
        let adj = &*ADJACENCY;
        let mut frontier = ArrayVec::new();
        let mut seen = [false; GRID_SIZE];

        for &tile_idx in &self.placed_tiles {
            for nidx in adj.neighbors_of(tile_idx as usize) {
                if !self.grid.get(nidx).is_present() && !seen[nidx] {
                    seen[nidx] = true;
                    if !frontier.is_full() {
                        frontier.push(nidx as u16);
                    }
                }
            }
        }
        frontier
    }

    /// Get positions where a specific wildlife can be placed.
    pub fn valid_wildlife_positions(&self, wildlife: Wildlife) -> ArrayVec<u16, 64> {
        let mut positions = ArrayVec::new();
        for &tile_idx in &self.placed_tiles {
            let cell = self.grid.get(tile_idx as usize);
            if cell.can_place_wildlife(wildlife) && !positions.is_full() {
                positions.push(tile_idx);
            }
        }
        positions
    }
}

impl Default for Board {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::WildlifeMask;

    fn forest_tile() -> TileData {
        TileData::single(Terrain::Forest, WildlifeMask::new(&[Wildlife::Bear]))
    }

    fn dual_tile() -> TileData {
        TileData::dual(
            Terrain::Forest,
            Terrain::River,
            WildlifeMask::new(&[Wildlife::Salmon, Wildlife::Hawk]),
        )
    }

    #[test]
    fn place_single_tile() {
        let mut board = Board::new();
        let action = board.place_tile(HexCoord::ORIGIN, forest_tile(), 0);
        assert!(action.is_some());
        assert_eq!(board.tile_count, 1);
        assert_eq!(board.largest_group[Terrain::Forest as usize], 1);
    }

    #[test]
    fn place_adjacent_same_terrain_unions() {
        let mut board = Board::new();
        board.place_tile(HexCoord::new(0, 0), forest_tile(), 0);
        board.place_tile(HexCoord::new(1, 0), forest_tile(), 0);
        assert_eq!(board.largest_group[Terrain::Forest as usize], 2);

        board.place_tile(HexCoord::new(0, 1), forest_tile(), 0);
        assert_eq!(board.largest_group[Terrain::Forest as usize], 3);
    }

    #[test]
    fn dual_terrain_tile() {
        let mut board = Board::new();
        board.place_tile(HexCoord::new(0, 0), dual_tile(), 0);
        assert_eq!(board.largest_group[Terrain::Forest as usize], 1);
        assert_eq!(board.largest_group[Terrain::River as usize], 1);
    }

    #[test]
    fn wildlife_placement() {
        let mut board = Board::new();
        board.place_tile(HexCoord::ORIGIN, forest_tile(), 0);
        let idx = HexCoord::ORIGIN.to_index().unwrap();

        // Can place bear (allowed)
        let action = board.place_wildlife(idx, Wildlife::Bear);
        assert!(action.is_some());
        assert_eq!(board.wildlife_positions[Wildlife::Bear as usize].len(), 1);

        // Nature token earned (keystone)
        assert_eq!(board.nature_tokens, 1);
    }

    #[test]
    fn wildlife_not_allowed() {
        let mut board = Board::new();
        board.place_tile(HexCoord::ORIGIN, forest_tile(), 0); // only bear allowed
        let idx = HexCoord::ORIGIN.to_index().unwrap();

        let action = board.place_wildlife(idx, Wildlife::Fox);
        assert!(action.is_none());
    }

    #[test]
    fn undo_tile_placement() {
        let mut board = Board::new();
        board.place_tile(HexCoord::new(0, 0), forest_tile(), 0);
        let action = board.place_tile(HexCoord::new(1, 0), forest_tile(), 0).unwrap();
        assert_eq!(board.largest_group[Terrain::Forest as usize], 2);
        assert_eq!(board.tile_count, 2);

        board.undo(action);
        assert_eq!(board.largest_group[Terrain::Forest as usize], 1);
        assert_eq!(board.tile_count, 1);
    }

    #[test]
    fn undo_wildlife_placement() {
        let mut board = Board::new();
        board.place_tile(HexCoord::ORIGIN, forest_tile(), 0);
        let idx = HexCoord::ORIGIN.to_index().unwrap();
        let action = board.place_wildlife(idx, Wildlife::Bear).unwrap();

        assert_eq!(board.nature_tokens, 1);
        board.undo(action);
        assert_eq!(board.nature_tokens, 0);
        assert_eq!(board.wildlife_positions[Wildlife::Bear as usize].len(), 0);
        assert!(!board.grid.get(idx).has_wildlife());
    }

    #[test]
    fn frontier_computation() {
        let mut board = Board::new();
        board.place_tile(HexCoord::ORIGIN, forest_tile(), 0);
        let frontier = board.frontier();
        assert_eq!(frontier.len(), 6); // origin has 6 empty neighbors
    }

    #[test]
    fn cannot_place_on_occupied() {
        let mut board = Board::new();
        board.place_tile(HexCoord::ORIGIN, forest_tile(), 0);
        let action = board.place_tile(HexCoord::ORIGIN, forest_tile(), 0);
        assert!(action.is_none());
    }
}
