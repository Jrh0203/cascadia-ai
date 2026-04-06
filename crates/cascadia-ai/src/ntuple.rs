use cascadia_core::board::Board;
use cascadia_core::hex::{HexCoord, GRID_SIZE};
use cascadia_core::types::{Terrain, Wildlife};

// ─────────────────────────────────────────────────────────────────────
// Cell encoding
// ─────────────────────────────────────────────────────────────────────

/// Wildlife cell encoding: 7 states.
///   0 = empty (no tile)
///   1-5 = bear, elk, salmon, hawk, fox
///   6 = tile present, no wildlife placed
const WILDLIFE_STATES: usize = 7;

/// Terrain cell encoding: 6 states.
///   0 = empty (no tile)
///   1-5 = forest, prairie, wetland, mountain, river
const TERRAIN_STATES: usize = 6;

#[inline(always)]
fn wildlife_cell(board: &Board, idx: usize) -> u8 {
    let cell = board.grid.get(idx);
    if !cell.is_present() {
        return 0;
    }
    match cell.placed_wildlife() {
        Some(w) => (w as u8) + 1, // bear=1, elk=2, salmon=3, hawk=4, fox=5
        None => 6,                 // tile present, no wildlife
    }
}

#[inline(always)]
fn terrain_cell(board: &Board, idx: usize) -> u8 {
    let cell = board.grid.get(idx);
    if !cell.is_present() {
        return 0;
    }
    match cell.primary_terrain() {
        Some(t) => (t as u8) + 1, // forest=1..river=5
        None => 0,
    }
}

// ─────────────────────────────────────────────────────────────────────
// Tuple index computation
// ─────────────────────────────────────────────────────────────────────

#[inline(always)]
fn pair_index(a: u8, b: u8, base: usize) -> usize {
    a as usize * base + b as usize
}

#[inline(always)]
fn triple_index(a: u8, b: u8, c: u8, base: usize) -> usize {
    (a as usize * base + b as usize) * base + c as usize
}

#[inline(always)]
fn quad_index(a: u8, b: u8, c: u8, d: u8, base: usize) -> usize {
    ((a as usize * base + b as usize) * base + c as usize) * base + d as usize
}

// ─────────────────────────────────────────────────────────────────────
// N-Tuple Network
// ─────────────────────────────────────────────────────────────────────

/// Table sizes
const WL_PAIR: usize = WILDLIFE_STATES * WILDLIFE_STATES;           // 49
const WL_LINE3: usize = WL_PAIR * WILDLIFE_STATES;                  // 343
const WL_LINE4: usize = WL_LINE3 * WILDLIFE_STATES;                 // 2401
const TR_PAIR: usize = TERRAIN_STATES * TERRAIN_STATES;             // 36
const TR_LINE3: usize = TR_PAIR * TERRAIN_STATES;                   // 216

const NUM_DIRS: usize = 3;

/// Total number of weights in the network.
pub const TOTAL_WEIGHTS: usize =
    NUM_DIRS * (WL_PAIR + WL_LINE3 + WL_LINE4 + TR_PAIR + TR_LINE3);

/// N-tuple network for board evaluation.
/// Uses line tuples (length 2, 3, 4) along 3 hex directions
/// with separate wildlife and terrain encodings.
#[derive(Clone)]
pub struct NTupleNetwork {
    /// Wildlife pair weights [direction][pair_index]
    wl_pair: [[f32; WL_PAIR]; NUM_DIRS],
    /// Wildlife line-3 weights
    wl_line3: [[f32; WL_LINE3]; NUM_DIRS],
    /// Wildlife line-4 weights
    wl_line4: [[f32; WL_LINE4]; NUM_DIRS],
    /// Terrain pair weights
    tr_pair: [[f32; TR_PAIR]; NUM_DIRS],
    /// Terrain line-3 weights
    tr_line3: [[f32; TR_LINE3]; NUM_DIRS],
}

impl NTupleNetwork {
    pub fn new() -> Self {
        NTupleNetwork {
            wl_pair: [[0.0; WL_PAIR]; NUM_DIRS],
            wl_line3: [[0.0; WL_LINE3]; NUM_DIRS],
            wl_line4: [[0.0; WL_LINE4]; NUM_DIRS],
            tr_pair: [[0.0; TR_PAIR]; NUM_DIRS],
            tr_line3: [[0.0; TR_LINE3]; NUM_DIRS],
        }
    }

    /// Scale all weights by a factor.
    pub fn scale(&mut self, factor: f32) {
        for dir in 0..NUM_DIRS {
            for w in self.wl_pair[dir].iter_mut() { *w *= factor; }
            for w in self.wl_line3[dir].iter_mut() { *w *= factor; }
            for w in self.wl_line4[dir].iter_mut() { *w *= factor; }
            for w in self.tr_pair[dir].iter_mut() { *w *= factor; }
            for w in self.tr_line3[dir].iter_mut() { *w *= factor; }
        }
    }

    /// Merge another network's weights into this one (element-wise addition).
    /// Used to accumulate TD updates from parallel workers.
    pub fn merge_from(&mut self, other: &NTupleNetwork) {
        for dir in 0..NUM_DIRS {
            for i in 0..WL_PAIR { self.wl_pair[dir][i] += other.wl_pair[dir][i]; }
            for i in 0..WL_LINE3 { self.wl_line3[dir][i] += other.wl_line3[dir][i]; }
            for i in 0..WL_LINE4 { self.wl_line4[dir][i] += other.wl_line4[dir][i]; }
            for i in 0..TR_PAIR { self.tr_pair[dir][i] += other.tr_pair[dir][i]; }
            for i in 0..TR_LINE3 { self.tr_line3[dir][i] += other.tr_line3[dir][i]; }
        }
    }

    /// Evaluate a board position. Returns estimated future value.
    pub fn evaluate(&self, board: &Board) -> f32 {
        let mut value = 0.0f32;

        // Iterate over placed tiles as starting points for tuples
        for &tile_idx in board.placed_tiles.iter() {
            let start = HexCoord::from_index(tile_idx as usize);

            for (dir, &(dq, dr)) in HexCoord::LINE_DIRECTIONS.iter().enumerate() {
                // Get cells along the line: start, start+dir, start+2*dir, start+3*dir
                let c0_idx = tile_idx as usize;
                let c1 = HexCoord::new(start.q + dq, start.r + dr);
                let c2 = HexCoord::new(start.q + 2 * dq, start.r + 2 * dr);
                let c3 = HexCoord::new(start.q + 3 * dq, start.r + 3 * dr);

                let w0 = wildlife_cell(board, c0_idx);
                let t0 = terrain_cell(board, c0_idx);

                // Line-2 (pair): always valid since c0 is placed
                if let Some(c1_idx) = c1.to_index() {
                    let w1 = wildlife_cell(board, c1_idx);
                    let t1 = terrain_cell(board, c1_idx);

                    value += self.wl_pair[dir][pair_index(w0, w1, WILDLIFE_STATES)];
                    value += self.tr_pair[dir][pair_index(t0, t1, TERRAIN_STATES)];

                    // Line-3
                    if let Some(c2_idx) = c2.to_index() {
                        let w2 = wildlife_cell(board, c2_idx);
                        let t2 = terrain_cell(board, c2_idx);

                        value += self.wl_line3[dir][triple_index(w0, w1, w2, WILDLIFE_STATES)];
                        value += self.tr_line3[dir][triple_index(t0, t1, t2, TERRAIN_STATES)];

                        // Line-4
                        if let Some(c3_idx) = c3.to_index() {
                            let w3 = wildlife_cell(board, c3_idx);
                            value += self.wl_line4[dir][quad_index(w0, w1, w2, w3, WILDLIFE_STATES)];
                        }
                    }
                }
            }
        }

        value
    }

    /// Count the number of tuples activated by this board state.
    fn count_tuples(&self, board: &Board) -> usize {
        let mut count = 0;
        for &tile_idx in board.placed_tiles.iter() {
            let start = HexCoord::from_index(tile_idx as usize);
            for (_, &(dq, dr)) in HexCoord::LINE_DIRECTIONS.iter().enumerate() {
                let c1 = HexCoord::new(start.q + dq, start.r + dr);
                if c1.to_index().is_some() {
                    count += 2; // wl_pair + tr_pair
                    let c2 = HexCoord::new(start.q + 2 * dq, start.r + 2 * dr);
                    if c2.to_index().is_some() {
                        count += 2; // wl_line3 + tr_line3
                        let c3 = HexCoord::new(start.q + 3 * dq, start.r + 3 * dr);
                        if c3.to_index().is_some() {
                            count += 1; // wl_line4
                        }
                    }
                }
            }
        }
        count
    }

    /// Collect all tuple indices activated by this board state.
    /// Returns a vec of (table_id, index) pairs for efficient batch updates.
    /// Table IDs: 0-2 = wl_pair dirs, 3-5 = wl_line3 dirs, 6-8 = wl_line4 dirs,
    ///            9-11 = tr_pair dirs, 12-14 = tr_line3 dirs
    fn activated_tuples(&self, board: &Board) -> Vec<(u8, u16)> {
        let mut tuples = Vec::with_capacity(board.placed_tiles.len() * 15);

        for &tile_idx in board.placed_tiles.iter() {
            let start = HexCoord::from_index(tile_idx as usize);

            for (dir, &(dq, dr)) in HexCoord::LINE_DIRECTIONS.iter().enumerate() {
                let c0_idx = tile_idx as usize;
                let c1 = HexCoord::new(start.q + dq, start.r + dr);
                let c2 = HexCoord::new(start.q + 2 * dq, start.r + 2 * dr);
                let c3 = HexCoord::new(start.q + 3 * dq, start.r + 3 * dr);

                let w0 = wildlife_cell(board, c0_idx);
                let t0 = terrain_cell(board, c0_idx);

                if let Some(c1_idx) = c1.to_index() {
                    let w1 = wildlife_cell(board, c1_idx);
                    let t1 = terrain_cell(board, c1_idx);

                    tuples.push((dir as u8, pair_index(w0, w1, WILDLIFE_STATES) as u16));
                    tuples.push((9 + dir as u8, pair_index(t0, t1, TERRAIN_STATES) as u16));

                    if let Some(c2_idx) = c2.to_index() {
                        let w2 = wildlife_cell(board, c2_idx);
                        let t2 = terrain_cell(board, c2_idx);

                        tuples.push((3 + dir as u8, triple_index(w0, w1, w2, WILDLIFE_STATES) as u16));
                        tuples.push((12 + dir as u8, triple_index(t0, t1, t2, TERRAIN_STATES) as u16));

                        if let Some(c3_idx) = c3.to_index() {
                            let w3 = wildlife_cell(board, c3_idx);
                            tuples.push((6 + dir as u8, quad_index(w0, w1, w2, w3, WILDLIFE_STATES) as u16));
                        }
                    }
                }
            }
        }

        tuples
    }

    /// TD update: adjust weights for all activated tuples by delta * alpha / num_tuples.
    /// Normalizes by tuple count so learning rate is independent of board size.
    pub fn update(&mut self, board: &Board, delta: f32, alpha: f32) {
        // Count activated tuples for normalization
        let num_tuples = self.count_tuples(board).max(1) as f32;
        let adjustment = delta * alpha / num_tuples;

        for &tile_idx in board.placed_tiles.iter() {
            let start = HexCoord::from_index(tile_idx as usize);

            for (dir, &(dq, dr)) in HexCoord::LINE_DIRECTIONS.iter().enumerate() {
                let c0_idx = tile_idx as usize;
                let c1 = HexCoord::new(start.q + dq, start.r + dr);
                let c2 = HexCoord::new(start.q + 2 * dq, start.r + 2 * dr);
                let c3 = HexCoord::new(start.q + 3 * dq, start.r + 3 * dr);

                let w0 = wildlife_cell(board, c0_idx);
                let t0 = terrain_cell(board, c0_idx);

                if let Some(c1_idx) = c1.to_index() {
                    let w1 = wildlife_cell(board, c1_idx);
                    let t1 = terrain_cell(board, c1_idx);

                    self.wl_pair[dir][pair_index(w0, w1, WILDLIFE_STATES)] += adjustment;
                    self.tr_pair[dir][pair_index(t0, t1, TERRAIN_STATES)] += adjustment;

                    if let Some(c2_idx) = c2.to_index() {
                        let w2 = wildlife_cell(board, c2_idx);
                        let t2 = terrain_cell(board, c2_idx);

                        self.wl_line3[dir][triple_index(w0, w1, w2, WILDLIFE_STATES)] += adjustment;
                        self.tr_line3[dir][triple_index(t0, t1, t2, TERRAIN_STATES)] += adjustment;

                        if let Some(c3_idx) = c3.to_index() {
                            let w3 = wildlife_cell(board, c3_idx);
                            self.wl_line4[dir][quad_index(w0, w1, w2, w3, WILDLIFE_STATES)] += adjustment;
                        }
                    }
                }
            }
        }
    }

    /// Save weights to a binary file.
    pub fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        use std::io::Write;
        let mut file = std::fs::File::create(path)?;

        // Write a simple header
        file.write_all(b"NTPL")?; // magic
        file.write_all(&1u32.to_le_bytes())?; // version

        // Write all weight tables
        for dir in 0..NUM_DIRS {
            for &w in &self.wl_pair[dir] {
                file.write_all(&w.to_le_bytes())?;
            }
            for &w in &self.wl_line3[dir] {
                file.write_all(&w.to_le_bytes())?;
            }
            for &w in &self.wl_line4[dir] {
                file.write_all(&w.to_le_bytes())?;
            }
            for &w in &self.tr_pair[dir] {
                file.write_all(&w.to_le_bytes())?;
            }
            for &w in &self.tr_line3[dir] {
                file.write_all(&w.to_le_bytes())?;
            }
        }

        Ok(())
    }

    /// Load weights from a binary file.
    pub fn load(path: &std::path::Path) -> std::io::Result<Self> {
        use std::io::Read;
        let mut file = std::fs::File::open(path)?;

        // Read header
        let mut magic = [0u8; 4];
        file.read_exact(&mut magic)?;
        if &magic != b"NTPL" {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "bad magic"));
        }
        let mut version = [0u8; 4];
        file.read_exact(&mut version)?;

        let mut net = NTupleNetwork::new();

        let mut buf = [0u8; 4];
        for dir in 0..NUM_DIRS {
            for w in net.wl_pair[dir].iter_mut() {
                file.read_exact(&mut buf)?;
                *w = f32::from_le_bytes(buf);
            }
            for w in net.wl_line3[dir].iter_mut() {
                file.read_exact(&mut buf)?;
                *w = f32::from_le_bytes(buf);
            }
            for w in net.wl_line4[dir].iter_mut() {
                file.read_exact(&mut buf)?;
                *w = f32::from_le_bytes(buf);
            }
            for w in net.tr_pair[dir].iter_mut() {
                file.read_exact(&mut buf)?;
                *w = f32::from_le_bytes(buf);
            }
            for w in net.tr_line3[dir].iter_mut() {
                file.read_exact(&mut buf)?;
                *w = f32::from_le_bytes(buf);
            }
        }

        Ok(net)
    }
}
