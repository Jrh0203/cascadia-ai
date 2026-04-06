use crate::types::Cell;

/// Grid dimension. 21×21 supports boards well beyond a 20-turn game.
pub const GRID_DIM: usize = 21;
pub const GRID_CENTER: i8 = 10;
pub const GRID_SIZE: usize = GRID_DIM * GRID_DIM; // 441

/// Axial hex coordinate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HexCoord {
    pub q: i8,
    pub r: i8,
}

impl HexCoord {
    pub const ORIGIN: HexCoord = HexCoord { q: 0, r: 0 };

    #[inline(always)]
    pub fn new(q: i8, r: i8) -> Self {
        HexCoord { q, r }
    }

    /// Convert to flat array index. Returns None if out of bounds.
    #[inline(always)]
    pub fn to_index(self) -> Option<usize> {
        let col = (self.q as i16) + (GRID_CENTER as i16);
        let row = (self.r as i16) + (GRID_CENTER as i16);
        if col < 0 || col >= GRID_DIM as i16 || row < 0 || row >= GRID_DIM as i16 {
            None
        } else {
            Some((col as usize) * GRID_DIM + (row as usize))
        }
    }

    /// Convert from flat array index.
    #[inline(always)]
    pub fn from_index(idx: usize) -> Self {
        let col = (idx / GRID_DIM) as i8 - GRID_CENTER;
        let row = (idx % GRID_DIM) as i8 - GRID_CENTER;
        HexCoord { q: col, r: row }
    }

    /// The 6 neighbor directions in axial coordinates.
    /// Order: E, NE, NW, W, SW, SE
    pub const DIRECTIONS: [(i8, i8); 6] = [
        (1, 0),   // E
        (1, -1),  // NE
        (0, -1),  // NW
        (-1, 0),  // W
        (-1, 1),  // SW
        (0, 1),   // SE
    ];

    /// The 3 line directions (each direction and its opposite form a line).
    /// E-W, NE-SW, NW-SE
    pub const LINE_DIRECTIONS: [(i8, i8); 3] = [
        (1, 0),   // E (opposite: W)
        (1, -1),  // NE (opposite: SW)
        (0, -1),  // NW (opposite: SE)
    ];

    #[inline(always)]
    pub fn neighbor(self, dir: usize) -> HexCoord {
        let (dq, dr) = Self::DIRECTIONS[dir];
        HexCoord {
            q: self.q + dq,
            r: self.r + dr,
        }
    }

    /// Get all valid neighbor indices.
    #[inline]
    pub fn neighbor_indices(self) -> impl Iterator<Item = usize> {
        Self::DIRECTIONS.iter().filter_map(move |&(dq, dr)| {
            HexCoord::new(self.q + dq, self.r + dr).to_index()
        })
    }
}

/// Precomputed adjacency lookup table.
/// For each grid position, stores the indices of its (up to 6) neighbors.
/// Uses u16::MAX as sentinel for "no neighbor" (out of bounds).
pub struct AdjacencyTable {
    pub neighbors: [[u16; 6]; GRID_SIZE],
    pub neighbor_count: [u8; GRID_SIZE],
}

impl AdjacencyTable {
    pub fn new() -> Self {
        let mut table = AdjacencyTable {
            neighbors: [[u16::MAX; 6]; GRID_SIZE],
            neighbor_count: [0; GRID_SIZE],
        };

        for idx in 0..GRID_SIZE {
            let coord = HexCoord::from_index(idx);
            let mut count = 0u8;
            for (dir, &(dq, dr)) in HexCoord::DIRECTIONS.iter().enumerate() {
                let nq = coord.q + dq;
                let nr = coord.r + dr;
                if let Some(nidx) = HexCoord::new(nq, nr).to_index() {
                    table.neighbors[idx][dir] = nidx as u16;
                    count += 1;
                }
            }
            table.neighbor_count[idx] = count;
        }

        table
    }

    /// Iterate valid neighbor indices for a given cell index.
    #[inline]
    pub fn neighbors_of(&self, idx: usize) -> NeighborIter<'_> {
        NeighborIter {
            data: &self.neighbors[idx],
            pos: 0,
        }
    }
}

pub struct NeighborIter<'a> {
    data: &'a [u16; 6],
    pos: usize,
}

impl<'a> Iterator for NeighborIter<'a> {
    type Item = usize;

    #[inline(always)]
    fn next(&mut self) -> Option<usize> {
        while self.pos < 6 {
            let val = self.data[self.pos];
            self.pos += 1;
            if val != u16::MAX {
                return Some(val as usize);
            }
        }
        None
    }
}

/// Static/lazy adjacency table singleton.
use std::sync::LazyLock;

pub static ADJACENCY: LazyLock<AdjacencyTable> = LazyLock::new(AdjacencyTable::new);

/// The hex grid: a flat array of cells.
#[derive(Clone)]
pub struct HexGrid {
    pub cells: [Cell; GRID_SIZE],
}

impl HexGrid {
    pub fn new() -> Self {
        HexGrid {
            cells: [Cell::EMPTY; GRID_SIZE],
        }
    }

    #[inline(always)]
    pub fn get(&self, idx: usize) -> Cell {
        self.cells[idx]
    }

    #[inline(always)]
    pub fn get_coord(&self, coord: HexCoord) -> Option<Cell> {
        coord.to_index().map(|idx| self.cells[idx])
    }

    #[inline(always)]
    pub fn set(&mut self, idx: usize, cell: Cell) {
        self.cells[idx] = cell;
    }

    #[inline(always)]
    pub fn set_coord(&mut self, coord: HexCoord, cell: Cell) {
        if let Some(idx) = coord.to_index() {
            self.cells[idx] = cell;
        }
    }
}

impl Default for HexGrid {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coord_index_roundtrip() {
        for q in -10..=10 {
            for r in -10..=10 {
                let c = HexCoord::new(q, r);
                let idx = c.to_index().unwrap();
                let c2 = HexCoord::from_index(idx);
                assert_eq!(c, c2);
            }
        }
    }

    #[test]
    fn coord_out_of_bounds() {
        assert!(HexCoord::new(11, 0).to_index().is_none());
        assert!(HexCoord::new(0, -11).to_index().is_none());
    }

    #[test]
    fn origin_has_six_neighbors() {
        let adj = &*ADJACENCY;
        let origin_idx = HexCoord::ORIGIN.to_index().unwrap();
        let count = adj.neighbors_of(origin_idx).count();
        assert_eq!(count, 6);
    }

    #[test]
    fn corner_has_fewer_neighbors() {
        let adj = &*ADJACENCY;
        let corner = HexCoord::new(-10, -10).to_index().unwrap();
        let count = adj.neighbors_of(corner).count();
        assert!(count < 6);
    }

    #[test]
    fn neighbor_directions_consistent() {
        // Neighbor in direction 0 (E) of origin should be (1, 0)
        let origin = HexCoord::ORIGIN;
        let east = origin.neighbor(0);
        assert_eq!(east, HexCoord::new(1, 0));

        // And (1,0)'s W neighbor (direction 3) should be origin
        let back = east.neighbor(3);
        assert_eq!(back, origin);
    }
}
