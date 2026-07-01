use arrayvec::ArrayVec;

use crate::hex::{HexCoord, HexGrid, ADJACENCY, GRID_SIZE};
use crate::types::{Cell, Terrain, TileData, Wildlife};

const PRIMARY_TERRAIN_EDGE_MASKS: [u8; 6] = [
    0b00_0111, 0b00_1110, 0b01_1100, 0b11_1000, 0b11_0001, 0b10_0011,
];

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

    #[inline]
    fn root_readonly(&self, mut x: usize) -> usize {
        while self.parent[x] as usize != x {
            x = self.parent[x] as usize;
        }
        x
    }

    #[inline]
    fn root_size_readonly(&self, root: usize) -> u16 {
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

#[derive(Debug, Clone, Copy)]
pub struct HabitatPreviewNeighbor {
    pub index: u16,
    pub terrain: Terrain,
    pub direction: u8,
}

#[derive(Debug, Clone)]
pub struct HabitatPreviewCell {
    pub index: u16,
    pub neighbors: ArrayVec<HabitatPreviewNeighbor, 6>,
}

#[derive(Debug, Clone)]
pub struct HabitatPreviewContext {
    pub base_total: u16,
    pub frontier: ArrayVec<HabitatPreviewCell, 128>,
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
    if offset < 3 {
        t1
    } else {
        t2
    }
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
    pub fn place_tile(
        &mut self,
        coord: HexCoord,
        tile: TileData,
        rotation: u8,
    ) -> Option<UndoAction> {
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

        let (habitat_merges, largest_group_before) = self.apply_tile_habitat(idx, cell, rotation);

        Some(UndoAction::PlaceTile {
            idx,
            habitat_merges,
            largest_group_before,
        })
    }

    fn apply_tile_habitat(
        &mut self,
        idx: usize,
        cell: Cell,
        rotation: u8,
    ) -> (ArrayVec<HabitatMerge, 12>, [u16; 5]) {
        let largest_before = self.largest_group;
        let mut merges: ArrayVec<HabitatMerge, 12> = ArrayVec::new();

        // Process each terrain on this tile
        let terrains: ArrayVec<Terrain, 2> = {
            let mut t = ArrayVec::new();
            t.push(
                cell.primary_terrain()
                    .expect("tile habitat application requires a present cell"),
            );
            if let Some(t2) = cell.secondary_terrain() {
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

        (merges, largest_before)
    }

    fn undo_tile_habitat(
        &mut self,
        idx: usize,
        cell: Cell,
        habitat_merges: &[HabitatMerge],
        largest_group_before: [u16; 5],
    ) {
        self.largest_group = largest_group_before;

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

        if let Some(terrain) = cell.primary_terrain() {
            self.habitat_uf[terrain as usize].size[idx] = 0;
        }
        if let Some(terrain) = cell.secondary_terrain() {
            self.habitat_uf[terrain as usize].size[idx] = 0;
        }
    }

    /// Reproduce the habitat-state side effects left by placing and undoing a
    /// tile without mutating the grid, tile stack, or counters.
    ///
    /// The legacy union-find undo intentionally restores merges but not path
    /// compression performed while evaluating a temporary tile. Search policy
    /// tie order therefore depends on those residual parent updates. This
    /// method uses the same habitat implementation as `place_tile` and `undo`
    /// while avoiding unrelated board work in exact-policy hot paths.
    pub fn replay_tile_place_undo_habitat_at_index(
        &mut self,
        idx: usize,
        tile: TileData,
        rotation: u8,
    ) -> bool {
        if idx >= GRID_SIZE || self.grid.get(idx).is_present() {
            return false;
        }

        let cell = tile.to_cell();
        let rotation = rotation % 6;
        let mut merges = ArrayVec::<(usize, [usize; 2], [u16; 2], [u8; 2]), 12>::new();
        let terrains = [cell.primary_terrain(), cell.secondary_terrain()];
        for terrain in terrains.into_iter().flatten() {
            let terrain_index = terrain as usize;
            let union_find = &mut self.habitat_uf[terrain_index];
            union_find.parent[idx] = idx as u16;
            union_find.rank[idx] = 0;

            for direction in 0..6 {
                let neighbor = ADJACENCY.neighbors[idx][direction];
                if neighbor == u16::MAX {
                    continue;
                }
                let neighbor = neighbor as usize;
                let neighbor_cell = self.grid.get(neighbor);
                if !neighbor_cell.is_present()
                    || terrain_on_edge(cell, rotation, direction) != Some(terrain)
                    || terrain_on_edge(neighbor_cell, self.rotations[neighbor], (direction + 3) % 6)
                        != Some(terrain)
                {
                    continue;
                }

                let root_a = union_find.find(idx);
                let root_b = union_find.find(neighbor);
                if root_a == root_b {
                    continue;
                }
                merges.push((
                    terrain_index,
                    [root_a, root_b],
                    [union_find.parent[root_a], union_find.parent[root_b]],
                    [union_find.rank[root_a], union_find.rank[root_b]],
                ));

                let (big, small) = if union_find.rank[root_a] >= union_find.rank[root_b] {
                    (root_a, root_b)
                } else {
                    (root_b, root_a)
                };
                union_find.parent[small] = big as u16;
                if union_find.rank[big] == union_find.rank[small] {
                    union_find.rank[big] += 1;
                }
            }

            // `place_tile` calls `set_size` after all merges. Its returned size
            // is discarded by this replay, but the final `find` can leave path
            // compression that the historical undo does not restore.
            union_find.find(idx);
        }

        for &(terrain_index, [root_a, root_b], parent_before, rank_before) in merges.iter().rev() {
            let union_find = &mut self.habitat_uf[terrain_index];
            union_find.parent[root_a] = parent_before[0];
            union_find.parent[root_b] = parent_before[1];
            union_find.rank[root_a] = rank_before[0];
            union_find.rank[root_b] = rank_before[1];
        }
        for terrain in terrains.into_iter().flatten() {
            self.habitat_uf[terrain as usize].size[idx] = 0;
        }
        self.grid.set(idx, Cell::EMPTY);
        self.rotations[idx] = 0;
        true
    }

    /// Prepared-neighbor form of
    /// [`Board::replay_tile_place_undo_habitat_at_index`].
    ///
    /// `HabitatPreviewCell` stores the occupied neighbors in the same
    /// direction order used by the full replay, together with the terrain
    /// facing this cell. Reusing it removes repeated grid, adjacency, and
    /// neighbor-edge decoding while preserving the legacy union/find order and
    /// its observable path-compression history exactly.
    pub fn replay_tile_place_undo_habitat_prepared(
        &mut self,
        preview: &HabitatPreviewCell,
        tile: TileData,
        rotation: u8,
    ) -> bool {
        let idx = preview.index as usize;
        if idx >= GRID_SIZE || self.grid.get(idx).is_present() {
            return false;
        }

        let cell = tile.to_cell();
        let rotation = rotation % 6;
        let primary = cell
            .primary_terrain()
            .expect("tile habitat replay requires a present cell");
        let secondary = cell.secondary_terrain();
        let primary_edge_mask = PRIMARY_TERRAIN_EDGE_MASKS[rotation as usize];
        let terrains = [Some(primary), secondary];
        let mut merges = ArrayVec::<(usize, [usize; 2], [u16; 2], [u8; 2]), 12>::new();

        for terrain in terrains.into_iter().flatten() {
            let terrain_index = terrain as usize;
            let union_find = &mut self.habitat_uf[terrain_index];
            union_find.parent[idx] = idx as u16;
            union_find.rank[idx] = 0;

            for neighbor in &preview.neighbors {
                let direction = neighbor.direction as usize;
                let edge_terrain =
                    if secondary.is_none() || primary_edge_mask & (1 << direction) != 0 {
                        primary
                    } else {
                        secondary.expect("a non-primary dual edge has secondary terrain")
                    };
                if edge_terrain != terrain || neighbor.terrain != terrain {
                    continue;
                }

                let root_a = union_find.find(idx);
                let root_b = union_find.find(neighbor.index as usize);
                if root_a == root_b {
                    continue;
                }
                merges.push((
                    terrain_index,
                    [root_a, root_b],
                    [union_find.parent[root_a], union_find.parent[root_b]],
                    [union_find.rank[root_a], union_find.rank[root_b]],
                ));

                let (big, small) = if union_find.rank[root_a] >= union_find.rank[root_b] {
                    (root_a, root_b)
                } else {
                    (root_b, root_a)
                };
                union_find.parent[small] = big as u16;
                if union_find.rank[big] == union_find.rank[small] {
                    union_find.rank[big] += 1;
                }
            }

            union_find.find(idx);
        }

        for &(terrain_index, [root_a, root_b], parent_before, rank_before) in merges.iter().rev() {
            let union_find = &mut self.habitat_uf[terrain_index];
            union_find.parent[root_a] = parent_before[0];
            union_find.parent[root_b] = parent_before[1];
            union_find.rank[root_a] = rank_before[0];
            union_find.rank[root_b] = rank_before[1];
        }
        for terrain in terrains.into_iter().flatten() {
            self.habitat_uf[terrain as usize].size[idx] = 0;
        }
        self.grid.set(idx, Cell::EMPTY);
        self.rotations[idx] = 0;
        true
    }

    /// Return the habitat-score total that `place_tile` would produce without
    /// mutating the board or its union-find structures.
    pub fn preview_habitat_total(
        &self,
        coord: HexCoord,
        tile: TileData,
        rotation: u8,
    ) -> Option<u16> {
        let idx = coord.to_index()?;
        self.preview_habitat_total_at_index(idx, tile, rotation)
    }

    /// Index-based form of [`Board::preview_habitat_total`] for hot action
    /// generation paths that already store frontier cells as flat indices.
    #[inline]
    pub fn preview_habitat_total_at_index(
        &self,
        idx: usize,
        tile: TileData,
        rotation: u8,
    ) -> Option<u16> {
        if idx >= GRID_SIZE {
            return None;
        }
        if self.grid.get(idx).is_present() {
            return None;
        }

        let rotation = rotation % 6;
        let base_total = self.largest_group.iter().sum::<u16>();
        let secondary = tile.terrain2.filter(|&terrain| terrain != tile.terrain1);
        let primary_edge_mask = PRIMARY_TERRAIN_EDGE_MASKS[rotation as usize];
        let mut roots = [ArrayVec::<u16, 6>::new(), ArrayVec::<u16, 6>::new()];
        let mut merged_sizes = [1u16; 2];
        let adj = &*ADJACENCY;
        for direction in 0..6 {
            let neighbor = adj.neighbors[idx][direction];
            if neighbor == u16::MAX {
                continue;
            }
            let neighbor = neighbor as usize;
            let neighbor_cell = self.grid.get(neighbor);
            if !neighbor_cell.is_present() {
                continue;
            }

            let secondary_edge = secondary.is_some() && primary_edge_mask & (1 << direction) == 0;
            let terrain_slot = usize::from(secondary_edge);
            let terrain = if secondary_edge {
                secondary.expect("secondary edge requires a secondary terrain")
            } else {
                tile.terrain1
            };
            if terrain_on_edge(neighbor_cell, self.rotations[neighbor], (direction + 3) % 6)
                != Some(terrain)
            {
                continue;
            }

            let terrain_index = terrain as usize;
            let root = self.habitat_uf[terrain_index].root_readonly(neighbor) as u16;
            if roots[terrain_slot].contains(&root) {
                continue;
            }
            roots[terrain_slot].push(root);
            merged_sizes[terrain_slot] +=
                self.habitat_uf[terrain_index].root_size_readonly(root as usize);
        }

        let primary_index = tile.terrain1 as usize;
        let mut total =
            base_total + merged_sizes[0].saturating_sub(self.largest_group[primary_index]);
        if let Some(secondary) = secondary {
            let secondary_index = secondary as usize;
            total += merged_sizes[1].saturating_sub(self.largest_group[secondary_index]);
        }
        Some(total)
    }

    /// Prepare the grid- and rotation-dependent portion of habitat previews.
    ///
    /// Rollout opponents evaluate the same frontier repeatedly while their
    /// legacy union-find parent pointers evolve through exact place/undo
    /// replay. Neighbor occupancy and edge terrain remain fixed, so retaining
    /// those immutable terms avoids recomputing them without caching any
    /// history-dependent habitat roots or sizes.
    pub fn habitat_preview_context(&self, frontier: &[u16]) -> HabitatPreviewContext {
        let adjacency = &*ADJACENCY;
        let mut prepared = ArrayVec::new();
        for &index in frontier {
            let idx = index as usize;
            if idx >= GRID_SIZE || self.grid.get(idx).is_present() || prepared.is_full() {
                continue;
            }

            let mut neighbors = ArrayVec::new();
            for direction in 0..6 {
                let neighbor = adjacency.neighbors[idx][direction];
                if neighbor == u16::MAX {
                    continue;
                }
                let neighbor_index = neighbor as usize;
                let neighbor_cell = self.grid.get(neighbor_index);
                if !neighbor_cell.is_present() {
                    continue;
                }
                let terrain = terrain_on_edge(
                    neighbor_cell,
                    self.rotations[neighbor_index],
                    (direction + 3) % 6,
                )
                .expect("a present neighboring tile has terrain on every edge");
                neighbors.push(HabitatPreviewNeighbor {
                    index: neighbor,
                    terrain,
                    direction: direction as u8,
                });
            }
            prepared.push(HabitatPreviewCell { index, neighbors });
        }

        HabitatPreviewContext {
            base_total: self.largest_group.iter().sum(),
            frontier: prepared,
        }
    }

    /// Evaluate one frontier cell from a prepared habitat-preview context.
    ///
    /// This still reads the current union-find roots on every call. That is
    /// required for bit-exact compatibility with the legacy policy, whose
    /// temporary place/undo sequence intentionally leaves path-compression
    /// side effects behind.
    #[inline]
    pub fn preview_habitat_total_prepared(
        &self,
        context: &HabitatPreviewContext,
        cell: &HabitatPreviewCell,
        tile: TileData,
        rotation: u8,
    ) -> u16 {
        let primary_index = tile.terrain1 as usize;
        let Some(secondary) = tile.terrain2.filter(|&terrain| terrain != tile.terrain1) else {
            let mut roots = [0u16; 6];
            let mut root_count = 0usize;
            let mut merged_size = 1u16;
            for neighbor in &cell.neighbors {
                if neighbor.terrain != tile.terrain1 {
                    continue;
                }
                let root =
                    self.habitat_uf[primary_index].root_readonly(neighbor.index as usize) as u16;
                if roots[..root_count].contains(&root) {
                    continue;
                }
                roots[root_count] = root;
                root_count += 1;
                merged_size += self.habitat_uf[primary_index].root_size_readonly(root as usize);
            }
            return context.base_total
                + merged_size.saturating_sub(self.largest_group[primary_index]);
        };

        let secondary_index = secondary as usize;
        let primary_edge_mask = PRIMARY_TERRAIN_EDGE_MASKS[(rotation % 6) as usize];
        let mut primary_roots = [0u16; 6];
        let mut secondary_roots = [0u16; 6];
        let mut primary_root_count = 0usize;
        let mut secondary_root_count = 0usize;
        let mut primary_size = 1u16;
        let mut secondary_size = 1u16;

        for neighbor in &cell.neighbors {
            if primary_edge_mask & (1 << neighbor.direction) != 0 {
                if neighbor.terrain != tile.terrain1 {
                    continue;
                }
                let root =
                    self.habitat_uf[primary_index].root_readonly(neighbor.index as usize) as u16;
                if primary_roots[..primary_root_count].contains(&root) {
                    continue;
                }
                primary_roots[primary_root_count] = root;
                primary_root_count += 1;
                primary_size += self.habitat_uf[primary_index].root_size_readonly(root as usize);
            } else {
                if neighbor.terrain != secondary {
                    continue;
                }
                let root =
                    self.habitat_uf[secondary_index].root_readonly(neighbor.index as usize) as u16;
                if secondary_roots[..secondary_root_count].contains(&root) {
                    continue;
                }
                secondary_roots[secondary_root_count] = root;
                secondary_root_count += 1;
                secondary_size +=
                    self.habitat_uf[secondary_index].root_size_readonly(root as usize);
            }
        }

        context.base_total
            + primary_size.saturating_sub(self.largest_group[primary_index])
            + secondary_size.saturating_sub(self.largest_group[secondary_index])
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
                let cell = self.grid.get(idx);
                self.undo_tile_habitat(idx, cell, &habitat_merges, largest_group_before);

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
    fn habitat_preview_matches_real_single_and_dual_placements() {
        let mut board = Board::new();
        board.place_tile(HexCoord::new(0, 0), forest_tile(), 0);
        board.place_tile(HexCoord::new(1, 0), dual_tile(), 2);
        board.place_tile(HexCoord::new(0, 1), forest_tile(), 0);

        let frontier = board.frontier();
        let context = board.habitat_preview_context(&frontier);
        for tile in [forest_tile(), dual_tile()] {
            let rotations = if tile.terrain2.is_some() { 6 } else { 1 };
            for (frontier_position, &frontier_index) in frontier.iter().enumerate() {
                let coord = HexCoord::from_index(frontier_index as usize);
                for rotation in 0..rotations {
                    let preview = board.preview_habitat_total(coord, tile, rotation).unwrap();
                    let prepared = board.preview_habitat_total_prepared(
                        &context,
                        &context.frontier[frontier_position],
                        tile,
                        rotation,
                    );
                    assert_eq!(prepared, preview);
                    let mut placed = board.clone();
                    placed.place_tile(coord, tile, rotation).unwrap();
                    assert_eq!(preview, placed.largest_group.iter().sum());
                }
            }
        }
    }

    #[test]
    fn prepared_habitat_preview_tracks_legacy_replay_history() {
        let allowed = WildlifeMask::new(&Wildlife::ALL);
        let tiles = [
            TileData::single(Terrain::Forest, allowed),
            TileData::dual(Terrain::Forest, Terrain::River, allowed),
            TileData::dual(Terrain::Prairie, Terrain::Wetland, allowed),
            TileData::dual(Terrain::Mountain, Terrain::Forest, allowed),
        ];
        let mut board = Board::new();
        board
            .place_tile(HexCoord::ORIGIN, tiles[0], 0)
            .expect("origin is empty");
        for turn in 0..12 {
            let frontier = board.frontier();
            let tile = tiles[(turn + 1) % tiles.len()];
            let rotation = if tile.terrain2.is_some() {
                (turn % 6) as u8
            } else {
                0
            };
            board
                .place_tile(
                    HexCoord::from_index(frontier[(turn * 5 + 1) % frontier.len()] as usize),
                    tile,
                    rotation,
                )
                .expect("selected frontier cell is empty");
        }

        let frontier = board.frontier();
        let context = board.habitat_preview_context(&frontier);
        for replay in 0..24 {
            let tile = tiles[replay % tiles.len()];
            let rotation = if tile.terrain2.is_some() {
                (replay % 6) as u8
            } else {
                0
            };
            let cell_position = (replay * 7 + 3) % context.frontier.len();
            let cell = &context.frontier[cell_position];

            let direct = board
                .preview_habitat_total_at_index(cell.index as usize, tile, rotation)
                .expect("prepared frontier cell stays empty");
            let prepared = board.preview_habitat_total_prepared(&context, cell, tile, rotation);
            assert_eq!(
                prepared, direct,
                "prepared preview mismatch before replay {replay}"
            );

            assert!(board.replay_tile_place_undo_habitat_at_index(
                cell.index as usize,
                tile,
                rotation,
            ));
        }

        for tile in tiles {
            let rotations = if tile.terrain2.is_some() { 6 } else { 1 };
            for cell in &context.frontier {
                for rotation in 0..rotations {
                    let direct = board
                        .preview_habitat_total_at_index(cell.index as usize, tile, rotation)
                        .expect("prepared frontier cell stays empty");
                    let prepared =
                        board.preview_habitat_total_prepared(&context, cell, tile, rotation);
                    assert_eq!(prepared, direct, "post-replay prepared preview mismatch");
                }
            }
        }
    }

    #[test]
    fn habitat_preview_matches_real_placements_across_evolving_board() {
        let allowed = WildlifeMask::new(&Wildlife::ALL);
        let tiles = [
            TileData::single(Terrain::Forest, allowed),
            TileData::single(Terrain::Mountain, allowed),
            TileData::dual(Terrain::Forest, Terrain::River, allowed),
            TileData::dual(Terrain::Prairie, Terrain::Wetland, allowed),
            TileData::dual(Terrain::Mountain, Terrain::Forest, allowed),
        ];
        let mut board = Board::new();
        board
            .place_tile(HexCoord::ORIGIN, tiles[0], 0)
            .expect("origin is empty");

        for turn in 0..19 {
            let frontier = board.frontier();
            for tile in tiles {
                let rotations = if tile.terrain2.is_some() { 6 } else { 1 };
                for &frontier_index in &frontier {
                    for rotation in 0..rotations {
                        let preview = board
                            .preview_habitat_total_at_index(frontier_index as usize, tile, rotation)
                            .expect("frontier cell is empty");
                        let mut placed = board.clone();
                        placed
                            .place_tile(
                                HexCoord::from_index(frontier_index as usize),
                                tile,
                                rotation,
                            )
                            .expect("frontier cell is empty");
                        assert_eq!(
                            preview,
                            placed.largest_group.iter().sum(),
                            "preview mismatch at turn {turn}, index {frontier_index}, rotation {rotation}"
                        );
                    }
                }
            }

            let tile = tiles[(turn + 1) % tiles.len()];
            let frontier_index = frontier[(turn * 7 + 3) % frontier.len()];
            let rotations = if tile.terrain2.is_some() { 6 } else { 1 };
            board
                .place_tile(
                    HexCoord::from_index(frontier_index as usize),
                    tile,
                    (turn % rotations) as u8,
                )
                .expect("selected frontier cell is empty");
        }
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
        let action = board
            .place_tile(HexCoord::new(1, 0), forest_tile(), 0)
            .unwrap();
        assert_eq!(board.largest_group[Terrain::Forest as usize], 2);
        assert_eq!(board.tile_count, 2);

        board.undo(action);
        assert_eq!(board.largest_group[Terrain::Forest as usize], 1);
        assert_eq!(board.tile_count, 1);
    }

    #[test]
    fn habitat_only_replay_matches_full_place_undo_state_exactly() {
        let allowed = WildlifeMask::new(&Wildlife::ALL);
        let tiles = [
            TileData::single(Terrain::Forest, allowed),
            TileData::single(Terrain::Mountain, allowed),
            TileData::dual(Terrain::Forest, Terrain::River, allowed),
            TileData::dual(Terrain::Prairie, Terrain::Wetland, allowed),
            TileData::dual(Terrain::Mountain, Terrain::Forest, allowed),
        ];
        let mut full = Board::new();
        full.place_tile(HexCoord::ORIGIN, tiles[0], 0).unwrap();
        for turn in 0..12 {
            let frontier = full.frontier();
            let tile = tiles[(turn + 1) % tiles.len()];
            let index = frontier[(turn * 5 + 1) % frontier.len()] as usize;
            full.place_tile(
                HexCoord::from_index(index),
                tile,
                (turn % if tile.terrain2.is_some() { 6 } else { 1 }) as u8,
            )
            .unwrap();
        }
        let mut replay = full.clone();
        let mut prepared_replay = full.clone();
        let prepared_frontier = full.frontier();
        let prepared_context = full.habitat_preview_context(&prepared_frontier);

        for tile in tiles {
            let rotations = if tile.terrain2.is_some() { 6 } else { 1 };
            let frontier = full.frontier();
            for &index in frontier.iter().take(12) {
                let preview = prepared_context
                    .frontier
                    .iter()
                    .find(|cell| cell.index == index)
                    .expect("prepared context contains every unchanged frontier cell");
                for rotation in 0..rotations {
                    let coord = HexCoord::from_index(index as usize);
                    let action = full.place_tile(coord, tile, rotation).unwrap();
                    full.undo(action);
                    assert!(replay.replay_tile_place_undo_habitat_at_index(
                        index as usize,
                        tile,
                        rotation,
                    ));
                    assert!(prepared_replay
                        .replay_tile_place_undo_habitat_prepared(preview, tile, rotation,));

                    assert_eq!(full.largest_group, replay.largest_group);
                    assert_eq!(full.largest_group, prepared_replay.largest_group);
                    assert_eq!(full.rotations, replay.rotations);
                    assert_eq!(full.rotations, prepared_replay.rotations);
                    assert_eq!(full.tile_count, replay.tile_count);
                    assert_eq!(full.tile_count, prepared_replay.tile_count);
                    assert_eq!(full.nature_tokens, replay.nature_tokens);
                    assert_eq!(full.nature_tokens, prepared_replay.nature_tokens);
                    assert_eq!(full.placed_tiles, replay.placed_tiles);
                    assert_eq!(full.placed_tiles, prepared_replay.placed_tiles);
                    assert_eq!(full.wildlife_positions, replay.wildlife_positions);
                    assert_eq!(full.wildlife_positions, prepared_replay.wildlife_positions);
                    for cell_index in 0..GRID_SIZE {
                        assert_eq!(full.grid.get(cell_index), replay.grid.get(cell_index));
                        assert_eq!(
                            full.grid.get(cell_index),
                            prepared_replay.grid.get(cell_index)
                        );
                    }
                    for terrain in 0..5 {
                        assert_eq!(
                            full.habitat_uf[terrain].parent,
                            replay.habitat_uf[terrain].parent
                        );
                        assert_eq!(
                            full.habitat_uf[terrain].rank,
                            replay.habitat_uf[terrain].rank
                        );
                        assert_eq!(
                            full.habitat_uf[terrain].size,
                            replay.habitat_uf[terrain].size
                        );
                        assert_eq!(
                            full.habitat_uf[terrain].parent,
                            prepared_replay.habitat_uf[terrain].parent
                        );
                        assert_eq!(
                            full.habitat_uf[terrain].rank,
                            prepared_replay.habitat_uf[terrain].rank
                        );
                        assert_eq!(
                            full.habitat_uf[terrain].size,
                            prepared_replay.habitat_uf[terrain].size
                        );
                    }
                }
            }
        }
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
