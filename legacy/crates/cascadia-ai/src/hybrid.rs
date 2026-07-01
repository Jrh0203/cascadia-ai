//! HybridNet: NNUE backbone + learnable HexCNN delta head.
//!
//! Architecture:  `v_final = v_nnue + alpha * Δ`
//!
//! - `v_nnue` is the existing NNUE evaluation (e.g. `nnue_weights_v4opp_modal_iter3`).
//! - `Δ` is a small HexCNN over a 16-plane 127-cell hex disk — captures
//!   patterns the NNUE feature set can't represent. ~33K params, ~0.5ms
//!   forward latency.
//! - `alpha` defaults to 0 so a fresh `HybridNetwork` is bit-identical to
//!   its inner `NNUENetwork`. Trained α settles around 0.2–0.3.
//!
//! The hex-disk topology, neighbor lookup, and SGEMM façade are all
//! shared with `alphazero_v2`.

use std::io::{self, Read, Write};
use std::sync::{LazyLock, OnceLock};

use rand::{rngs::StdRng, Rng, SeedableRng};

use cascadia_core::board::Board;
use cascadia_core::hex::{AdjacencyTable, HexCoord};

use crate::nnue::{v6_peak, NNUENetwork};
use crate::sgemm::sgemm_rm;

// ─────────────────────────────────────────────────────────────────────
// Compact 127-cell hex disk topology. Mirrors the `alphazero_v2`
// constants and `hex_neighbors_local` so this module compiles with any
// feature-flag set (in particular, without `az-v2` / `v6-peak`).
// ─────────────────────────────────────────────────────────────────────

/// Number of real cells in the 127-cell hex disk (V6_LOCAL_RADIUS = 6).
pub const AZ_LOCAL_CELLS: usize = 127;
/// Pad cell index — out-of-disk neighbors route here so they stay inert
/// under HexConv.
pub const AZ_PAD_INDEX: usize = 127;
/// Total cells including pad: `127 + 1`.
pub const AZ_CELLS_PADDED: usize = 128;

const GRID_DIM: usize = 21;

static HEX_NEIGHBORS_LOCAL: OnceLock<Vec<[usize; 7]>> = OnceLock::new();

fn build_hex_neighbors_local() -> Vec<[usize; 7]> {
    let mut out = vec![[AZ_PAD_INDEX; 7]; AZ_CELLS_PADDED];
    // Hex 6-direction offsets in (dcol, drow) — matches cascadia_core::hex.
    const DIRS: [(i32, i32); 6] = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)];
    for li in 0..AZ_LOCAL_CELLS {
        let g = v6_peak::local_to_global(li) as usize;
        let col = (g / GRID_DIM) as i32;
        let row = (g % GRID_DIM) as i32;
        let mut row7 = [AZ_PAD_INDEX; 7];
        row7[0] = li; // self
        for (k, (dc, dr)) in DIRS.iter().enumerate() {
            let nc = col + dc;
            let nr = row + dr;
            row7[1 + k] =
                if (0..GRID_DIM as i32).contains(&nc) && (0..GRID_DIM as i32).contains(&nr) {
                    let ng = (nc as usize) * GRID_DIM + (nr as usize);
                    let nl = v6_peak::global_to_local(ng);
                    if nl >= 0 {
                        nl as usize
                    } else {
                        AZ_PAD_INDEX
                    }
                } else {
                    AZ_PAD_INDEX
                };
        }
        out[li] = row7;
    }
    out
}

/// Per local cell [0, 128), returns [self, n0..n5] local indices.
/// Out-of-disk neighbors → `AZ_PAD_INDEX`. Pad cell at index 127 has all
/// 7 entries = `AZ_PAD_INDEX` so it stays inert under convolution.
pub fn hex_neighbors_local() -> &'static [[usize; 7]] {
    HEX_NEIGHBORS_LOCAL.get_or_init(build_hex_neighbors_local)
}

// ─────────────────────────────────────────────────────────────────────
// Architecture constants
// ─────────────────────────────────────────────────────────────────────

// v2 architecture — richer planes + bigger trunk.
// Layout grew from 16 → 61 planes (added 6-dir wildlife adjacency + habitat
// cluster role per terrain). Trunk widened 32→64 ch, blocks 2→3, hidden 16→32.
// Total params: ~33K → ~215K. Forward latency target: ≤ 0.5ms.
pub const DELTA_INPUT_CHANNELS: usize = 61;
pub const DELTA_TRUNK_CHANNELS: usize = 64;
pub const DELTA_BLOCKS: usize = 3;
pub const DELTA_HIDDEN: usize = 32;

/// Plane indices. Documented so trainers / tests can probe specific channels.
pub const DELTA_PLANE_BIAS: usize = 0;
pub const DELTA_PLANE_PRESENT: usize = 1;
pub const DELTA_PLANE_KEYSTONE: usize = 2;
/// 5 planes: Bear, Elk, Salmon, Hawk, Fox (Wildlife::from_u8 order).
pub const DELTA_PLANE_WILDLIFE_START: usize = 3;
/// 5 planes: Forest, Prairie, Wetland, Mountain, River (Terrain::from_u8 order).
pub const DELTA_PLANE_TERRAIN_START: usize = 8;
pub const DELTA_PLANE_RECENCY: usize = 13;
pub const DELTA_PLANE_DIST_FRONTIER: usize = 14;
pub const DELTA_PLANE_IS_FRONTIER: usize = 15;
/// 30 planes: per-cell × 6-direction × wildlife. For each cell c and direction
/// d, plane (16 + d*5 + w) fires when c's neighbor in direction d has placed
/// wildlife w. Captures pattern signal NNUE encodes sparsely; here it's dense
/// for the CNN to learn over.
pub const DELTA_PLANE_WL_ADJ_START: usize = 16; // 16..46 (30 planes)
/// 15 planes: per-cell × terrain × habitat-cluster-role (largest / 2nd / other).
/// Uses `nnue::v6_peak::compute_habitat_roles`. Captures Stockfish-style
/// cluster-membership conditioning that pattern features in NNUE already use.
pub const DELTA_PLANE_HAB_ROLE_START: usize = 46; // 46..61 (15 planes)

/// File-format magic. AZR3 = AlphaZero Rust v3 (HybridNet = NNUE + Δ).
pub const HYBRID_MAGIC: &[u8; 4] = b"AZR3";

/// Static bias plane: 1.0 on real cells [0, 127), 0 on pad [127].
static BIAS_PLANE: LazyLock<[f32; AZ_CELLS_PADDED]> = LazyLock::new(|| {
    let mut p = [0.0f32; AZ_CELLS_PADDED];
    for li in 0..AZ_LOCAL_CELLS {
        p[li] = 1.0;
    }
    p
});

static ADJACENCY: LazyLock<AdjacencyTable> = LazyLock::new(AdjacencyTable::new);

// ─────────────────────────────────────────────────────────────────────
// HexConv (sgemm-backed). Mirrors the design of `alphazero_v2::HexConv`
// at smaller channel counts for the Δ trunk.
// ─────────────────────────────────────────────────────────────────────

#[derive(Clone)]
struct HexConv {
    in_c: usize,
    out_c: usize,
    /// row-major `[out_c, in_c * 7]`
    w: Vec<f32>,
    b: Vec<f32>,
}

impl HexConv {
    fn new(in_c: usize, out_c: usize, rng: &mut StdRng) -> Self {
        let scale = (2.0 / (in_c * 7) as f32).sqrt();
        let w: Vec<f32> = (0..out_c * in_c * 7)
            .map(|_| rng.gen_range(-1.0..1.0) * scale)
            .collect();
        HexConv {
            in_c,
            out_c,
            w,
            b: vec![0.0; out_c],
        }
    }

    fn forward(&self, input: &[f32]) -> Vec<f32> {
        let neighbors = hex_neighbors_local();
        let in_c = self.in_c;
        let out_c = self.out_c;
        let cells = AZ_CELLS_PADDED;
        let k_dim = in_c * 7;

        // im2col: `cols[(ic*7 + k) * 128 + cell] = input[ic * 128 + neighbors[cell][k]]`
        let mut cols = vec![0.0f32; k_dim * cells];
        for ic in 0..in_c {
            let ib = ic * cells;
            for k in 0..7 {
                let row_base = (ic * 7 + k) * cells;
                for cell in 0..cells {
                    cols[row_base + cell] = input[ib + neighbors[cell][k]];
                }
            }
        }
        let mut out = vec![0.0f32; out_c * cells];
        sgemm_rm(out_c, cells, k_dim, 1.0, &self.w, &cols, 0.0, &mut out);
        // Broadcast bias per output channel.
        for oc in 0..out_c {
            let bias = self.b[oc];
            let ob = oc * cells;
            for cell in 0..cells {
                out[ob + cell] += bias;
            }
        }
        out
    }

    fn read<R: Read>(r: &mut R) -> io::Result<Self> {
        let in_c = read_u32(r)? as usize;
        let out_c = read_u32(r)? as usize;
        let w = read_vec_f32(r, out_c * in_c * 7)?;
        let b = read_vec_f32(r, out_c)?;
        Ok(HexConv { in_c, out_c, w, b })
    }

    fn write<W: Write>(&self, w: &mut W) -> io::Result<()> {
        write_u32(w, self.in_c as u32)?;
        write_u32(w, self.out_c as u32)?;
        write_vec_f32(w, &self.w)?;
        write_vec_f32(w, &self.b)?;
        Ok(())
    }
}

#[derive(Clone)]
struct ResHexBlock {
    c1: HexConv,
    c2: HexConv,
}

impl ResHexBlock {
    fn new(channels: usize, rng: &mut StdRng) -> Self {
        ResHexBlock {
            c1: HexConv::new(channels, channels, rng),
            c2: HexConv::new(channels, channels, rng),
        }
    }

    fn forward(&self, input: &[f32]) -> Vec<f32> {
        let mut z1 = self.c1.forward(input);
        for v in z1.iter_mut() {
            *v = v.max(0.0);
        }
        let z2 = self.c2.forward(&z1);
        let mut out = vec![0.0f32; z2.len()];
        for i in 0..out.len() {
            out[i] = (z2[i] + input[i]).max(0.0);
        }
        out
    }

    fn read<R: Read>(r: &mut R) -> io::Result<Self> {
        let c1 = HexConv::read(r)?;
        let c2 = HexConv::read(r)?;
        Ok(ResHexBlock { c1, c2 })
    }

    fn write<W: Write>(&self, w: &mut W) -> io::Result<()> {
        self.c1.write(w)?;
        self.c2.write(w)?;
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────
// DeltaNet — 16 planes → HexConv stem → 2 ResHexBlocks → pool → MLP → Δ
// ─────────────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct DeltaNet {
    pub input_channels: usize,
    pub trunk_channels: usize,
    pub blocks: usize,
    pub hidden: usize,
    stem: HexConv,
    block_stack: Vec<ResHexBlock>,
    /// `[hidden, trunk_channels]` row-major
    head_w1: Vec<f32>,
    head_b1: Vec<f32>,
    /// `[hidden]`
    head_w2: Vec<f32>,
    head_b2: f32,
}

impl DeltaNet {
    pub fn new(seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let in_c = DELTA_INPUT_CHANNELS;
        let trunk_c = DELTA_TRUNK_CHANNELS;
        let n_blocks = DELTA_BLOCKS;
        let hidden = DELTA_HIDDEN;
        let stem = HexConv::new(in_c, trunk_c, &mut rng);
        let block_stack: Vec<ResHexBlock> = (0..n_blocks)
            .map(|_| ResHexBlock::new(trunk_c, &mut rng))
            .collect();
        let scale_h1 = (2.0 / trunk_c as f32).sqrt();
        let head_w1: Vec<f32> = (0..hidden * trunk_c)
            .map(|_| rng.gen_range(-1.0..1.0) * scale_h1)
            .collect();
        let head_b1 = vec![0.0; hidden];
        // Zero init on the final scalar so a fresh DeltaNet returns Δ=0.
        // Combined with alpha=0, this means a fresh HybridNetwork is exactly
        // its inner NNUE — and even if alpha is bumped to non-zero before
        // training, Δ=0 keeps behavior unchanged. The downstream trainer
        // breaks this symmetry by gradient descent on residuals.
        let head_w2 = vec![0.0; hidden];
        let head_b2 = 0.0;
        DeltaNet {
            input_channels: in_c,
            trunk_channels: trunk_c,
            blocks: n_blocks,
            hidden,
            stem,
            block_stack,
            head_w1,
            head_b1,
            head_w2,
            head_b2,
        }
    }

    /// Forward pass: input is `[input_channels * AZ_CELLS_PADDED]` row-major
    /// (channel-major: index = ch * 128 + cell).
    pub fn forward(&self, input: &[f32]) -> f32 {
        debug_assert_eq!(input.len(), self.input_channels * AZ_CELLS_PADDED);
        let mut x = self.stem.forward(input);
        for v in x.iter_mut() {
            *v = v.max(0.0);
        }
        for block in &self.block_stack {
            x = block.forward(&x);
        }
        // Mean-pool over real cells only (exclude pad index).
        let c = self.trunk_channels;
        let mut pooled = vec![0.0f32; c];
        for ch in 0..c {
            let mut s = 0.0;
            for cell in 0..AZ_LOCAL_CELLS {
                s += x[ch * AZ_CELLS_PADDED + cell];
            }
            pooled[ch] = s / AZ_LOCAL_CELLS as f32;
        }
        // Head: trunk_c → hidden → 1.
        let mut h = self.head_b1.clone();
        for i in 0..self.hidden {
            let row_base = i * c;
            for j in 0..c {
                h[i] += self.head_w1[row_base + j] * pooled[j];
            }
            if h[i] < 0.0 {
                h[i] = 0.0;
            }
        }
        let mut out = self.head_b2;
        for i in 0..self.hidden {
            out += self.head_w2[i] * h[i];
        }
        out
    }

    fn read<R: Read>(r: &mut R) -> io::Result<Self> {
        let input_channels = read_u32(r)? as usize;
        let trunk_channels = read_u32(r)? as usize;
        let blocks = read_u32(r)? as usize;
        let hidden = read_u32(r)? as usize;
        let stem = HexConv::read(r)?;
        let mut block_stack = Vec::with_capacity(blocks);
        for _ in 0..blocks {
            block_stack.push(ResHexBlock::read(r)?);
        }
        let head_w1 = read_vec_f32(r, hidden * trunk_channels)?;
        let head_b1 = read_vec_f32(r, hidden)?;
        let head_w2 = read_vec_f32(r, hidden)?;
        let head_b2 = read_f32(r)?;
        Ok(DeltaNet {
            input_channels,
            trunk_channels,
            blocks,
            hidden,
            stem,
            block_stack,
            head_w1,
            head_b1,
            head_w2,
            head_b2,
        })
    }

    fn write<W: Write>(&self, w: &mut W) -> io::Result<()> {
        write_u32(w, self.input_channels as u32)?;
        write_u32(w, self.trunk_channels as u32)?;
        write_u32(w, self.blocks as u32)?;
        write_u32(w, self.hidden as u32)?;
        self.stem.write(w)?;
        for b in &self.block_stack {
            b.write(w)?;
        }
        write_vec_f32(w, &self.head_w1)?;
        write_vec_f32(w, &self.head_b1)?;
        write_vec_f32(w, &self.head_w2)?;
        write_f32(w, self.head_b2)?;
        Ok(())
    }

    /// Total parameter count — sanity check at construction.
    pub fn param_count(&self) -> usize {
        let stem_p = self.stem.w.len() + self.stem.b.len();
        let block_p: usize = self
            .block_stack
            .iter()
            .map(|b| b.c1.w.len() + b.c1.b.len() + b.c2.w.len() + b.c2.b.len())
            .sum();
        let head_p = self.head_w1.len() + self.head_b1.len() + self.head_w2.len() + 1;
        stem_p + block_p + head_p
    }
}

// ─────────────────────────────────────────────────────────────────────
// Board → 16-plane compact encoder.
// Layout: channel-major (`input[ch * 128 + cell]`), pad cell at index 127.
// ─────────────────────────────────────────────────────────────────────

pub fn encode_board_compact(board: &Board) -> Vec<f32> {
    let mut input = vec![0.0f32; DELTA_INPUT_CHANNELS * AZ_CELLS_PADDED];

    // Plane 0: bias (1.0 on real cells, 0 on pad).
    {
        let bias = &*BIAS_PLANE;
        input[..AZ_CELLS_PADDED].copy_from_slice(bias);
    }

    // Walk every present global cell once; map to its local index; emit
    // wildlife / terrain / keystone / present planes.
    let total_placed = board.placed_tiles.len().max(1) as f32;
    let mut recency_for_local = [0.0f32; AZ_LOCAL_CELLS];
    for (stack_pos, &tile_idx) in board.placed_tiles.iter().enumerate() {
        let global = tile_idx as usize;
        let local = v6_peak::global_to_local(global);
        if local < 0 {
            // Off-disk placement (rare: ~0.1%). Skip; the trunk has no
            // notion of this cell.
            continue;
        }
        let local = local as usize;
        let cell = board.grid.get(global);

        // Plane 1: present.
        input[1 * AZ_CELLS_PADDED + local] = 1.0;
        // Plane 2: keystone.
        if cell.is_keystone() {
            input[2 * AZ_CELLS_PADDED + local] = 1.0;
        }
        // Planes 3..8: wildlife one-hot.
        if let Some(w) = cell.placed_wildlife() {
            let wi = w as usize;
            if wi < 5 {
                input[(DELTA_PLANE_WILDLIFE_START + wi) * AZ_CELLS_PADDED + local] = 1.0;
            }
        }
        // Planes 8..13: primary terrain one-hot. (Secondary terrain on dual
        // tiles is captured by the per-direction terrain-on-edge block in
        // NNUE; DeltaNet stays simple here.)
        if let Some(t) = cell.primary_terrain() {
            let ti = t as usize;
            if ti < 5 {
                input[(DELTA_PLANE_TERRAIN_START + ti) * AZ_CELLS_PADDED + local] = 1.0;
            }
        }
        // Plane 13: recency = (stack_pos + 1) / total_placed.
        recency_for_local[local] = (stack_pos + 1) as f32 / total_placed;
    }
    for local in 0..AZ_LOCAL_CELLS {
        input[DELTA_PLANE_RECENCY * AZ_CELLS_PADDED + local] = recency_for_local[local];
    }

    // Frontier plane (plane 15): empty cells adjacent to ≥1 placed.
    // We also derive distance-to-frontier (plane 14) from the frontier set
    // via Chebyshev distance (clamped + normalized by 6).
    let frontier = board.frontier();
    let adj = &*ADJACENCY;
    // Plane 15: is-frontier (binary).
    for &fi in frontier.iter() {
        let local = v6_peak::global_to_local(fi as usize);
        if local >= 0 {
            input[DELTA_PLANE_IS_FRONTIER * AZ_CELLS_PADDED + local as usize] = 1.0;
        }
    }

    // Plane 14: distance-to-frontier. For each local cell, compute hex
    // distance to nearest frontier cell. Capped at 6, normalized to [0,1].
    // Uses the existing AdjacencyTable BFS (one pass from the frontier
    // outward, marking each local cell with the BFS depth).
    let mut dist = [u8::MAX; 441]; // global → distance
    let mut queue: Vec<u16> = Vec::with_capacity(frontier.len() * 2);
    for &fi in frontier.iter() {
        dist[fi as usize] = 0;
        queue.push(fi);
    }
    let mut head = 0usize;
    while head < queue.len() {
        let g = queue[head] as usize;
        head += 1;
        let d = dist[g];
        if d >= 6 {
            continue;
        }
        for n in adj.neighbors_of(g) {
            if dist[n] == u8::MAX {
                dist[n] = d + 1;
                queue.push(n as u16);
            }
        }
    }
    for local in 0..AZ_LOCAL_CELLS {
        let g = v6_peak::local_to_global(local) as usize;
        let d = dist[g];
        let norm = if d == u8::MAX {
            1.0
        } else {
            (d as f32 / 6.0).min(1.0)
        };
        input[DELTA_PLANE_DIST_FRONTIER * AZ_CELLS_PADDED + local] = norm;
    }

    // Planes 16..46 (30 planes): per-cell × 6-direction × wildlife adjacency.
    // For each present cell c, plane (16 + d*5 + w) fires if c's neighbor in
    // direction d (HexCoord::DIRECTIONS) has placed wildlife `w`. This gives
    // the CNN dense access to pattern signals (bear pairs, elk lines, salmon
    // runs, hawk LOS) that NNUE encodes sparsely.
    for local in 0..AZ_LOCAL_CELLS {
        let g = v6_peak::local_to_global(local) as usize;
        let cell = board.grid.get(g);
        if !cell.is_present() {
            continue;
        }
        // Translate global index → axial coord → enumerate 6 directions.
        let coord = HexCoord::from_index(g);
        for d in 0..6 {
            let n_coord = coord.neighbor(d);
            let n_global = match n_coord.to_index() {
                Some(idx) => idx,
                None => continue,
            };
            let n_cell = board.grid.get(n_global);
            if !n_cell.is_present() {
                continue;
            }
            if let Some(wl) = n_cell.placed_wildlife() {
                let wi = wl as usize;
                if wi < 5 {
                    let ch = DELTA_PLANE_WL_ADJ_START + d * 5 + wi;
                    input[ch * AZ_CELLS_PADDED + local] = 1.0;
                }
            }
        }
    }

    // Planes 46..61 (15 planes): per-cell × terrain × habitat cluster role.
    // For each cell that participates in a terrain-t cluster, fires one of
    // 3 planes: largest cluster / 2nd-largest cluster / other. Reuses the
    // existing `nnue::v6_peak::compute_habitat_roles` (proven BFS impl).
    let hab_roles = v6_peak::compute_habitat_roles(board);
    for local in 0..AZ_LOCAL_CELLS {
        let g = v6_peak::local_to_global(local) as usize;
        for t in 0..5 {
            if let Some(role) = hab_roles[g][t] {
                let role = role as usize;
                if role < 3 {
                    let ch = DELTA_PLANE_HAB_ROLE_START + t * 3 + role;
                    input[ch * AZ_CELLS_PADDED + local] = 1.0;
                }
            }
        }
    }

    input
}

// ─────────────────────────────────────────────────────────────────────
// HybridNetwork — wraps NNUE + DeltaNet + alpha
// ─────────────────────────────────────────────────────────────────────

pub struct HybridNetwork {
    pub nnue: NNUENetwork,
    pub delta: DeltaNet,
    pub alpha: f32,
}

impl HybridNetwork {
    /// Build a HybridNetwork from a trained NNUE plus a fresh (zero-output)
    /// Δ head. Alpha defaults to 0 → predictions identical to the inner
    /// NNUE. Training breaks this by gradient descent on score residuals.
    pub fn from_nnue(nnue: NNUENetwork, delta_seed: u64) -> Self {
        HybridNetwork {
            nnue,
            delta: DeltaNet::new(delta_seed),
            alpha: 0.0,
        }
    }

    /// Total predicted score. `features` are the NNUE sparse-feature
    /// indices already extracted for `board`; pass them in to avoid
    /// re-extracting from inside this function (the caller usually has
    /// them anyway). For the Δ path we encode the board directly.
    ///
    /// Fast path: if `alpha == 0.0`, skip the Δ computation entirely.
    pub fn evaluate(&self, board: &Board, features: &[u16]) -> f32 {
        let v_nnue = self.nnue.forward(features);
        if self.alpha == 0.0 {
            return v_nnue;
        }
        let input = encode_board_compact(board);
        let delta = self.delta.forward(&input);
        v_nnue + self.alpha * delta
    }

    /// Evaluate exposing both components — useful for debugging / training
    /// diagnostics where we want the residual breakdown.
    pub fn evaluate_components(&self, board: &Board, features: &[u16]) -> (f32, f32) {
        let v_nnue = self.nnue.forward(features);
        let delta = if self.alpha == 0.0 {
            0.0
        } else {
            let input = encode_board_compact(board);
            self.delta.forward(&input)
        };
        (v_nnue, delta)
    }

    /// Save only the Δ + α to `azr3_path`. The NNUE weights are saved
    /// independently via `NNUENetwork::save(nnue_path)` — keeps the existing
    /// NNUE binary format untouched. Two-file convention:
    ///   `champion_hybrid.azr3`  ← α + Δ (this file, small ~150KB)
    ///   `champion_hybrid.nnue`  ← standard NNUE (~23MB)
    ///
    /// The trainer writes both files; inference loads both.
    pub fn save_delta(&self, azr3_path: &std::path::Path) -> io::Result<()> {
        let mut f = std::fs::File::create(azr3_path)?;
        f.write_all(HYBRID_MAGIC)?;
        write_f32(&mut f, self.alpha)?;
        write_u32(&mut f, 1)?; // delta-format version
        self.delta.write(&mut f)?;
        Ok(())
    }

    /// Load a Δ+α file and attach a pre-loaded NNUE. Returns the assembled
    /// HybridNetwork. Use this when you already have an NNUE in memory
    /// (e.g. via `NNUENetwork::load(nnue_path)`).
    pub fn load_with_nnue(azr3_path: &std::path::Path, nnue: NNUENetwork) -> io::Result<Self> {
        let mut f = std::fs::File::open(azr3_path)?;
        let mut magic = [0u8; 4];
        f.read_exact(&mut magic)?;
        if &magic != HYBRID_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("hybrid: bad magic, expected AZR3, got {:?}", magic),
            ));
        }
        let alpha = read_f32(&mut f)?;
        let _version = read_u32(&mut f)?;
        let delta = DeltaNet::read(&mut f)?;
        Ok(HybridNetwork { nnue, delta, alpha })
    }

    /// Convenience: load both files together. Equivalent to
    /// `NNUENetwork::load(nnue_path)` followed by `Self::load_with_nnue`.
    pub fn load_paired(
        azr3_path: &std::path::Path,
        nnue_path: &std::path::Path,
    ) -> io::Result<Self> {
        let nnue = NNUENetwork::load(nnue_path)?;
        Self::load_with_nnue(azr3_path, nnue)
    }
}

// ─────────────────────────────────────────────────────────────────────
// HYBR — residual training data format.
//
// Layout:
//   Magic       : b"HYBR"        4 bytes
//   Version     : u32 (= 1)      4 bytes
//   Channels    : u32 (= 16)     4 bytes
//   Cells       : u32 (= 128)    4 bytes
//   <records...> (read until EOF)
//
// Per record:
//   board       : [f32; 16 * 128]    8192 bytes (channel-major)
//   nnue_pred   : f32                  4 bytes  — what NNUE predicted for this afterstate
//   label       : f32                  4 bytes  — true "remaining points" = final − current
//
// The Python trainer computes the residual as `label − nnue_pred` and
// regresses Δ to that. Storing both halves separately (instead of just
// the residual) lets the trainer also compute calibrated end-to-end RMSE
// against the true label.
// ─────────────────────────────────────────────────────────────────────

pub const HYBR_MAGIC: &[u8; 4] = b"HYBR";
pub const HYBR_VERSION: u32 = 1;

/// Single training record. `board` is `DELTA_INPUT_CHANNELS * AZ_CELLS_PADDED` floats.
pub struct HybridResidualRecord {
    pub board: Vec<f32>,
    pub nnue_pred: f32,
    pub label: f32,
}

/// Append records to a HYBR file. Writes the header iff the file doesn't
/// already exist (mirroring the `mce_policy_samples` append pattern).
pub fn append_hybrid_residuals(
    path: &std::path::Path,
    records: &[HybridResidualRecord],
) -> io::Result<()> {
    let is_new = !path.exists();
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    if is_new {
        f.write_all(HYBR_MAGIC)?;
        write_u32(&mut f, HYBR_VERSION)?;
        write_u32(&mut f, DELTA_INPUT_CHANNELS as u32)?;
        write_u32(&mut f, AZ_CELLS_PADDED as u32)?;
    }
    let expected_board_len = DELTA_INPUT_CHANNELS * AZ_CELLS_PADDED;
    for rec in records {
        if rec.board.len() != expected_board_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "HYBR record: board has len {}, expected {}",
                    rec.board.len(),
                    expected_board_len
                ),
            ));
        }
        write_vec_f32(&mut f, &rec.board)?;
        write_f32(&mut f, rec.nnue_pred)?;
        write_f32(&mut f, rec.label)?;
    }
    Ok(())
}

/// Load all records from a HYBR file.
pub fn load_hybrid_residuals(path: &std::path::Path) -> io::Result<Vec<HybridResidualRecord>> {
    let mut f = std::fs::File::open(path)?;
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic)?;
    if &magic != HYBR_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("HYBR: bad magic, got {:?}", magic),
        ));
    }
    let version = read_u32(&mut f)?;
    if version != HYBR_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "HYBR: unsupported version {} (expected {})",
                version, HYBR_VERSION
            ),
        ));
    }
    let channels = read_u32(&mut f)? as usize;
    let cells = read_u32(&mut f)? as usize;
    if channels != DELTA_INPUT_CHANNELS || cells != AZ_CELLS_PADDED {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "HYBR: shape {}x{} doesn't match DeltaNet input {}x{}",
                channels, cells, DELTA_INPUT_CHANNELS, AZ_CELLS_PADDED
            ),
        ));
    }
    let board_len = channels * cells;
    let mut out = Vec::new();
    loop {
        let mut board = vec![0.0f32; board_len];
        let mut buf = [0u8; 4];
        // Try to read the first f32 of a new record. EOF here = clean end.
        match f.read_exact(&mut buf) {
            Ok(()) => board[0] = f32::from_le_bytes(buf),
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e),
        }
        for i in 1..board_len {
            f.read_exact(&mut buf)?;
            board[i] = f32::from_le_bytes(buf);
        }
        let nnue_pred = read_f32(&mut f)?;
        let label = read_f32(&mut f)?;
        out.push(HybridResidualRecord {
            board,
            nnue_pred,
            label,
        });
    }
    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────
// HYBP — pairwise-discrimination training data.
//
// One record = one DECISION, holding all K candidates' (board, nnue_pred,
// mce_estimated_value). The trainer samples pairs (a, b) within each
// decision and learns Δ to predict the conditional residual difference:
//
//   target_AB = (mce_value_A - nnue_pred_A) - (mce_value_B - nnue_pred_B)
//   pred_AB   = delta(board_A) - delta(board_B)
//
// The constant residual bias (~+7) cancels in pairwise differences, so Δ
// is forced to learn discrimination signal, not calibration.
//
// Layout:
//   Magic       : b"HYBP"        4 bytes
//   Version     : u32 (= 1)      4 bytes
//   Channels    : u32 (= 16)     4 bytes
//   Cells       : u32 (= 128)    4 bytes
//   <decisions...> (read until EOF)
//
// Per decision:
//   K           : u32                   4 bytes — number of candidates
//   K × candidate record:
//     board       : [f32; 16 * 128]    8192 bytes
//     nnue_pred   : f32                  4 bytes
//     mce_value   : f32                  4 bytes
// ─────────────────────────────────────────────────────────────────────

pub const HYBP_MAGIC: &[u8; 4] = b"HYBP";
pub const HYBP_VERSION: u32 = 1;

pub struct HybridPairwiseDecision {
    /// Per-candidate (board, nnue_pred, mce_value).
    pub candidates: Vec<(Vec<f32>, f32, f32)>,
}

pub fn append_hybrid_pairwise(
    path: &std::path::Path,
    decisions: &[HybridPairwiseDecision],
) -> io::Result<()> {
    let is_new = !path.exists();
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    if is_new {
        f.write_all(HYBP_MAGIC)?;
        write_u32(&mut f, HYBP_VERSION)?;
        write_u32(&mut f, DELTA_INPUT_CHANNELS as u32)?;
        write_u32(&mut f, AZ_CELLS_PADDED as u32)?;
    }
    let board_len = DELTA_INPUT_CHANNELS * AZ_CELLS_PADDED;
    for dec in decisions {
        write_u32(&mut f, dec.candidates.len() as u32)?;
        for (board, nnue_pred, mce_value) in &dec.candidates {
            if board.len() != board_len {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "HYBP: bad board len",
                ));
            }
            write_vec_f32(&mut f, board)?;
            write_f32(&mut f, *nnue_pred)?;
            write_f32(&mut f, *mce_value)?;
        }
    }
    Ok(())
}

pub fn load_hybrid_pairwise(path: &std::path::Path) -> io::Result<Vec<HybridPairwiseDecision>> {
    let mut f = std::fs::File::open(path)?;
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic)?;
    if &magic != HYBP_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("HYBP: bad magic, got {:?}", magic),
        ));
    }
    let version = read_u32(&mut f)?;
    if version != HYBP_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("HYBP: unsupported version {}", version),
        ));
    }
    let channels = read_u32(&mut f)? as usize;
    let cells = read_u32(&mut f)? as usize;
    if channels != DELTA_INPUT_CHANNELS || cells != AZ_CELLS_PADDED {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "HYBP: shape mismatch",
        ));
    }
    let board_len = channels * cells;
    let mut out = Vec::new();
    loop {
        let mut k_buf = [0u8; 4];
        match f.read_exact(&mut k_buf) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e),
        }
        let k = u32::from_le_bytes(k_buf) as usize;
        let mut candidates: Vec<(Vec<f32>, f32, f32)> = Vec::with_capacity(k);
        for _ in 0..k {
            let mut board = vec![0.0f32; board_len];
            let mut buf = [0u8; 4];
            for i in 0..board_len {
                f.read_exact(&mut buf)?;
                board[i] = f32::from_le_bytes(buf);
            }
            let nnue_pred = read_f32(&mut f)?;
            let mce_value = read_f32(&mut f)?;
            candidates.push((board, nnue_pred, mce_value));
        }
        out.push(HybridPairwiseDecision { candidates });
    }
    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────
// Serialization helpers — small and self-contained.
// ─────────────────────────────────────────────────────────────────────

fn write_u32<W: Write>(w: &mut W, v: u32) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn read_u32<R: Read>(r: &mut R) -> io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn write_f32<W: Write>(w: &mut W, v: f32) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn read_f32<R: Read>(r: &mut R) -> io::Result<f32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(f32::from_le_bytes(b))
}

fn write_vec_f32<W: Write>(w: &mut W, v: &[f32]) -> io::Result<()> {
    for x in v {
        write_f32(w, *x)?;
    }
    Ok(())
}

fn read_vec_f32<R: Read>(r: &mut R, n: usize) -> io::Result<Vec<f32>> {
    let mut out = Vec::with_capacity(n);
    let mut buf = [0u8; 4];
    for _ in 0..n {
        r.read_exact(&mut buf)?;
        out.push(f32::from_le_bytes(buf));
    }
    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use cascadia_core::game::GameState;
    use cascadia_core::types::ScoringCards;

    fn fresh_game(seed: u64) -> GameState {
        let mut rng = StdRng::seed_from_u64(seed);
        GameState::new(4, ScoringCards::all_a(), &mut rng)
    }

    /// Plane 0 is 1.0 on real cells, 0 on pad. The constant LazyLock must be
    /// the same instance across calls (Stage 0.8.A-style static).
    #[test]
    fn bias_plane_static_and_correct() {
        let g = fresh_game(0xA001);
        let input = encode_board_compact(&g.boards[0]);
        for cell in 0..AZ_LOCAL_CELLS {
            assert_eq!(input[cell], 1.0, "bias cell {} should be 1.0", cell);
        }
        assert_eq!(input[AZ_PAD_INDEX], 0.0, "bias plane pad cell must be 0.0");
    }

    /// Encoder output has the right shape and the pad cell is zero on every
    /// non-bias plane.
    #[test]
    fn encode_board_compact_shape_and_pad() {
        let g = fresh_game(0xA002);
        let input = encode_board_compact(&g.boards[0]);
        assert_eq!(input.len(), DELTA_INPUT_CHANNELS * AZ_CELLS_PADDED);
        for ch in 1..DELTA_INPUT_CHANNELS {
            let idx = ch * AZ_CELLS_PADDED + AZ_PAD_INDEX;
            assert_eq!(input[idx], 0.0, "pad cell on plane {} should be 0", ch);
        }
    }

    /// DeltaNet with default zero-initialized output weights returns Δ = 0
    /// for any input. This is the property that makes a fresh HybridNetwork
    /// bit-identical to its inner NNUE (without needing alpha=0 explicitly).
    #[test]
    fn delta_net_zero_init_returns_zero() {
        let net = DeltaNet::new(0xB001);
        let g = fresh_game(0xB002);
        let input = encode_board_compact(&g.boards[0]);
        let delta = net.forward(&input);
        assert_eq!(delta, 0.0, "fresh DeltaNet should return 0, got {}", delta);
    }

    /// With non-zero output weights (simulating a trained Δ) the forward
    /// pass produces a finite, bounded scalar.
    #[test]
    fn delta_net_random_output_is_finite() {
        let mut net = DeltaNet::new(0xB101);
        let mut rng = StdRng::seed_from_u64(0xB102);
        // Hand-perturb the output layer to break the zero invariant.
        for v in net.head_w2.iter_mut() {
            *v = rng.gen_range(-0.5..0.5);
        }
        net.head_b2 = 0.1;
        let g = fresh_game(0xB103);
        let input = encode_board_compact(&g.boards[0]);
        let delta = net.forward(&input);
        assert!(delta.is_finite(), "Δ must be finite, got {}", delta);
        // Bounded — the head should be O(1) at the start of training.
        assert!(delta.abs() < 50.0, "|Δ| should be bounded, got {}", delta);
    }

    /// HybridNetwork.evaluate with alpha=0 is bit-identical to the inner
    /// NNUE's forward(). This is the property that lets us deploy a
    /// HybridNetwork as a champion with zero risk: until Δ is trained
    /// and alpha ramped, it just IS the NNUE.
    #[test]
    fn hybrid_alpha_zero_equals_nnue() {
        let nnue = NNUENetwork::new();
        let hybrid = HybridNetwork::from_nnue(nnue.clone(), 0xC001);
        assert_eq!(hybrid.alpha, 0.0, "fresh HybridNetwork should have alpha=0");

        let g = fresh_game(0xC002);
        let features = crate::nnue::extract_features(&g.boards[0]);
        let v_nnue = nnue.forward(&features);
        let v_hybrid = hybrid.evaluate(&g.boards[0], &features);
        assert_eq!(
            v_nnue, v_hybrid,
            "alpha=0 must yield bit-identical NNUE prediction"
        );
    }

    /// Same as above but with alpha != 0 and Δ identically zero (the post-
    /// `new()` invariant). Predictions must still match the NNUE bit-for-bit.
    #[test]
    fn hybrid_zero_delta_equals_nnue_at_any_alpha() {
        let nnue = NNUENetwork::new();
        let mut hybrid = HybridNetwork::from_nnue(nnue.clone(), 0xC101);
        hybrid.alpha = 0.3;

        let g = fresh_game(0xC102);
        let features = crate::nnue::extract_features(&g.boards[0]);
        let v_nnue = nnue.forward(&features);
        let v_hybrid = hybrid.evaluate(&g.boards[0], &features);
        assert_eq!(
            v_nnue, v_hybrid,
            "zero-init Δ × any α must still equal NNUE"
        );
    }

    /// Param-count sanity: v2 arch ~215K total (61→64ch stem + 3×64ch
    /// ResBlocks + 64→32→1 head). Catch accidental arch regressions.
    #[test]
    fn delta_net_param_count_matches_spec() {
        let net = DeltaNet::new(0xD001);
        let n = net.param_count();
        // Stem: 64 * (61*7) + 64 = 27_392
        // Block × 3 × 2 HexConv × (64*7*64 + 64) = 3 × 2 × 28_736 = 172_416
        // Head: 32*64 + 32 + 32 + 1 = 2_113
        // Total: ~201_921.
        assert!(
            (180_000..240_000).contains(&n),
            "DeltaNet param count {} outside expected band [180k, 240k]",
            n
        );
    }

    /// Recency monotonicity: the k-th tile placed has recency proportional
    /// to (k+1)/total_placed. The most recent placement has recency closest
    /// to 1.0; the oldest is smallest.
    #[test]
    fn recency_plane_monotone_in_placement_order() {
        // Run a real game forward a few turns to build up a non-trivial
        // placed_tiles stack.
        let mut g = fresh_game(0xE001);
        for _ in 0..6 {
            if g.is_game_over() {
                break;
            }
            // Make any legal move; rely on greedy default.
            match crate::search::greedy_move(&g) {
                Some(mv) => {
                    if !crate::search::execute_scored_move(&mut g, &mv) {
                        break;
                    }
                }
                None => break,
            }
        }
        let board = &g.boards[0];
        if board.placed_tiles.len() < 2 {
            return; // game progressed differently; skip silently
        }
        let input = encode_board_compact(board);
        // Find recency for first vs last placed.
        let first_global = board.placed_tiles[0] as usize;
        let last_global = *board.placed_tiles.last().unwrap() as usize;
        let first_local = v6_peak::global_to_local(first_global);
        let last_local = v6_peak::global_to_local(last_global);
        if first_local < 0 || last_local < 0 {
            return; // off-disk; skip
        }
        let recency_first = input[DELTA_PLANE_RECENCY * AZ_CELLS_PADDED + first_local as usize];
        let recency_last = input[DELTA_PLANE_RECENCY * AZ_CELLS_PADDED + last_local as usize];
        assert!(
            recency_last > recency_first,
            "last-placed recency ({}) should exceed first-placed ({})",
            recency_last,
            recency_first
        );
    }

    /// HYBR file round-trip: append a few records, load them back, verify
    /// the binary layout survives. Catches accidental schema drift.
    #[test]
    fn hybr_format_roundtrip() {
        let tmp = std::env::temp_dir().join("hybr_roundtrip_test.hybr");
        let _ = std::fs::remove_file(&tmp);

        let board_len = DELTA_INPUT_CHANNELS * AZ_CELLS_PADDED;
        let mut records = Vec::new();
        for i in 0..5 {
            let mut board = vec![0.0f32; board_len];
            // Distinguishable payload per record.
            for (j, v) in board.iter_mut().enumerate() {
                *v = (i as f32) * 0.01 + (j as f32) * 0.0001;
            }
            records.push(HybridResidualRecord {
                board,
                nnue_pred: 10.0 + i as f32,
                label: 12.5 + i as f32,
            });
        }
        append_hybrid_residuals(&tmp, &records).expect("append");
        let loaded = load_hybrid_residuals(&tmp).expect("load");
        assert_eq!(loaded.len(), records.len());
        for (a, b) in records.iter().zip(loaded.iter()) {
            assert_eq!(a.nnue_pred, b.nnue_pred);
            assert_eq!(a.label, b.label);
            assert_eq!(a.board.len(), b.board.len());
            for (x, y) in a.board.iter().zip(b.board.iter()) {
                assert_eq!(*x, *y);
            }
        }

        // A second append should preserve previously-written records.
        append_hybrid_residuals(&tmp, &records[..2]).expect("append again");
        let loaded2 = load_hybrid_residuals(&tmp).expect("load again");
        assert_eq!(loaded2.len(), records.len() + 2);
        let _ = std::fs::remove_file(&tmp);
    }

    /// HYBP (pairwise) format round-trip.
    #[test]
    fn hybp_format_roundtrip() {
        let tmp = std::env::temp_dir().join("hybp_roundtrip_test.hybp");
        let _ = std::fs::remove_file(&tmp);

        let board_len = DELTA_INPUT_CHANNELS * AZ_CELLS_PADDED;
        let mut decisions = Vec::new();
        for d in 0..3u32 {
            let mut cands = Vec::new();
            let k = 2 + d as usize; // varying K per decision
            for c in 0..k {
                let mut board = vec![0.0f32; board_len];
                for (j, v) in board.iter_mut().enumerate() {
                    *v = (d as f32) * 0.1 + (c as f32) * 0.01 + (j as f32) * 0.0001;
                }
                cands.push((board, 10.0 + c as f32, 11.0 + c as f32 * 0.5));
            }
            decisions.push(HybridPairwiseDecision { candidates: cands });
        }
        append_hybrid_pairwise(&tmp, &decisions).expect("append");
        let loaded = load_hybrid_pairwise(&tmp).expect("load");
        assert_eq!(loaded.len(), decisions.len());
        for (a, b) in decisions.iter().zip(loaded.iter()) {
            assert_eq!(a.candidates.len(), b.candidates.len());
            for ((ba, na, va), (bb, nb, vb)) in a.candidates.iter().zip(b.candidates.iter()) {
                assert_eq!(na, nb);
                assert_eq!(va, vb);
                for (x, y) in ba.iter().zip(bb.iter()) {
                    assert_eq!(x, y);
                }
            }
        }
        let _ = std::fs::remove_file(&tmp);
    }

    /// Save → load round-trip for the Δ+α file. The NNUE is provided
    /// separately on load (two-file format).
    #[test]
    fn hybrid_delta_save_load_roundtrip() {
        let nnue_a = NNUENetwork::new();
        let nnue_b = nnue_a.clone();
        let mut hybrid = HybridNetwork::from_nnue(nnue_a, 0xF001);
        hybrid.alpha = 0.25;

        let tmp = std::env::temp_dir().join("hybrid_delta_roundtrip_test.azr3");
        hybrid.save_delta(&tmp).expect("save");
        let loaded = HybridNetwork::load_with_nnue(&tmp, nnue_b).expect("load");
        assert_eq!(hybrid.alpha, loaded.alpha);
        assert_eq!(hybrid.delta.input_channels, loaded.delta.input_channels);
        assert_eq!(hybrid.delta.trunk_channels, loaded.delta.trunk_channels);

        let g = fresh_game(0xF002);
        let features = crate::nnue::extract_features(&g.boards[0]);
        let v_orig = hybrid.evaluate(&g.boards[0], &features);
        let v_load = loaded.evaluate(&g.boards[0], &features);
        assert_eq!(v_orig, v_load);
        let _ = std::fs::remove_file(&tmp);
    }
}
