//! AlphaZero-style residual CNN + PUCT pipeline for Cascadia.
//!
//! This is intentionally separate from the NNUE/CZero pipeline. It uses a dense
//! board tensor, residual convolutions, a spatial policy head, and a scalar value
//! head. The policy is evaluated over the legal candidate set by factorizing each
//! candidate into tile-cell, wildlife-cell, market-slot, and wildlife-market-slot
//! logits.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::cmp::Ordering;
use std::sync::OnceLock;

use cascadia_core::game::GameState;
use cascadia_core::hex::{HexCoord, GRID_DIM, GRID_SIZE};
use cascadia_core::scoring::ScoreBreakdown;
use cascadia_core::types::{ScoringCards, Wildlife};

use crate::eval::ScoredMove;
use crate::search::{execute_scored_move, greedy_move};

pub const AZ_INPUT_CHANNELS: usize = 65;
pub const AZ_VALUE_SCALE: f32 = 120.0;
const AZ_MAGIC: &[u8; 4] = b"AZR1";
const AZ_DATA_MAGIC: &[u8; 4] = b"AZD1";
static CONV_NEIGHBORS: OnceLock<Vec<Vec<(usize, usize)>>> = OnceLock::new();

#[derive(Debug, Clone, Copy)]
pub struct AlphaZeroConfig {
    pub channels: usize,
    pub blocks: usize,
    pub value_hidden: usize,
    pub max_candidates: usize,
    pub c_puct: f32,
}

impl Default for AlphaZeroConfig {
    fn default() -> Self {
        AlphaZeroConfig {
            channels: 16,
            blocks: 2,
            value_hidden: 64,
            max_candidates: 24,
            c_puct: 2.0,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct TrainStats {
    pub samples: usize,
    pub policy_loss: f32,
    pub value_loss: f32,
}

#[derive(Debug, Clone)]
pub struct AzSample {
    pub input: Vec<f32>,
    pub candidates: Vec<ScoredMove>,
    pub policy: Vec<f32>,
    pub value: f32,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AzDataSummary {
    pub samples: usize,
    pub input_channels: usize,
    pub grid_dim: usize,
    pub grid_size: usize,
    pub max_candidates: usize,
}

#[derive(Clone)]
struct Conv2d {
    in_c: usize,
    out_c: usize,
    w: Vec<f32>,
    b: Vec<f32>,
}

impl Conv2d {
    fn new(in_c: usize, out_c: usize, rng: &mut StdRng) -> Self {
        let fan = (in_c * 9).max(1) as f32;
        let scale = (2.0 / fan).sqrt();
        let mut w = Vec::with_capacity(out_c * in_c * 9);
        for _ in 0..out_c * in_c * 9 {
            w.push(rng.gen_range(-scale..scale));
        }
        Conv2d {
            in_c,
            out_c,
            w,
            b: vec![0.0; out_c],
        }
    }

    fn forward(&self, input: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0; self.out_c * GRID_SIZE];
        let neighbors = conv_neighbors();
        for oc in 0..self.out_c {
            for idx in 0..GRID_SIZE {
                let mut sum = self.b[oc];
                for ic in 0..self.in_c {
                    let wbase = (oc * self.in_c + ic) * 9;
                    let ibase = ic * GRID_SIZE;
                    for &(k, nidx) in &neighbors[idx] {
                        sum += self.w[wbase + k] * input[ibase + nidx];
                    }
                }
                out[oc * GRID_SIZE + idx] = sum;
            }
        }
        out
    }

    fn backward_update(
        &mut self,
        input: &[f32],
        grad_out: &[f32],
        grad_input: &mut [f32],
        lr: f32,
    ) {
        let neighbors = conv_neighbors();
        for oc in 0..self.out_c {
            let mut gb = 0.0f32;
            for idx in 0..GRID_SIZE {
                let go = grad_out[oc * GRID_SIZE + idx];
                gb += go;
                for ic in 0..self.in_c {
                    let wbase = (oc * self.in_c + ic) * 9;
                    let ibase = ic * GRID_SIZE;
                    for &(k, nidx) in &neighbors[idx] {
                        let wi = wbase + k;
                        let ii = ibase + nidx;
                        let old_w = self.w[wi];
                        grad_input[ii] += go * old_w;
                        self.w[wi] -= lr * go * input[ii];
                    }
                }
            }
            self.b[oc] -= lr * gb;
        }
    }
}

#[derive(Clone)]
struct ResidualBlock {
    c1: Conv2d,
    c2: Conv2d,
}

impl ResidualBlock {
    fn new(channels: usize, rng: &mut StdRng) -> Self {
        ResidualBlock {
            c1: Conv2d::new(channels, channels, rng),
            c2: Conv2d::new(channels, channels, rng),
        }
    }
}

struct BlockCache {
    input: Vec<f32>,
    z1: Vec<f32>,
    a1: Vec<f32>,
    pre: Vec<f32>,
}

struct ForwardCache {
    input: Vec<f32>,
    stem_z: Vec<f32>,
    blocks: Vec<BlockCache>,
    trunk: Vec<f32>,
    pooled: Vec<f32>,
    vh_pre: Vec<f32>,
    vh: Vec<f32>,
    value: f32,
    tile_logits: Vec<f32>,
    wildlife_logits: Vec<f32>,
    market_logits: [f32; 4],
    wildlife_market_logits: [f32; 4],
    skip_logit: f32,
}

#[derive(Clone)]
pub struct AlphaZeroNet {
    cfg: AlphaZeroConfig,
    stem: Conv2d,
    blocks: Vec<ResidualBlock>,
    policy_tile_w: Vec<f32>,
    policy_tile_b: f32,
    policy_wildlife_w: Vec<f32>,
    policy_wildlife_b: f32,
    policy_market_w: Vec<f32>,
    policy_market_b: [f32; 4],
    policy_wildlife_market_w: Vec<f32>,
    policy_wildlife_market_b: [f32; 4],
    policy_skip_w: Vec<f32>,
    policy_skip_b: f32,
    value_w1: Vec<f32>,
    value_b1: Vec<f32>,
    value_w2: Vec<f32>,
    value_b2: f32,
}

impl AlphaZeroNet {
    pub fn new(cfg: AlphaZeroConfig, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let channels = cfg.channels;
        let stem = Conv2d::new(AZ_INPUT_CHANNELS, channels, &mut rng);
        let blocks = (0..cfg.blocks)
            .map(|_| ResidualBlock::new(channels, &mut rng))
            .collect();
        let head_scale = (2.0 / channels.max(1) as f32).sqrt();
        AlphaZeroNet {
            cfg,
            stem,
            blocks,
            policy_tile_w: rand_vec(&mut rng, channels, head_scale),
            policy_tile_b: 0.0,
            policy_wildlife_w: rand_vec(&mut rng, channels, head_scale),
            policy_wildlife_b: 0.0,
            policy_market_w: rand_vec(&mut rng, 4 * channels, head_scale),
            policy_market_b: [0.0; 4],
            policy_wildlife_market_w: rand_vec(&mut rng, 4 * channels, head_scale),
            policy_wildlife_market_b: [0.0; 4],
            policy_skip_w: rand_vec(&mut rng, channels, head_scale),
            policy_skip_b: 0.0,
            value_w1: rand_vec(&mut rng, channels * cfg.value_hidden, head_scale),
            value_b1: vec![0.0; cfg.value_hidden],
            value_w2: rand_vec(
                &mut rng,
                cfg.value_hidden,
                (2.0 / cfg.value_hidden.max(1) as f32).sqrt(),
            ),
            value_b2: 0.0,
        }
    }

    pub fn config(&self) -> AlphaZeroConfig {
        self.cfg
    }

    fn forward_cache(&self, input: &[f32]) -> ForwardCache {
        let stem_z = self.stem.forward(input);
        let stem_a = relu_vec(&stem_z);
        let mut x = stem_a.clone();
        let mut blocks = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            let z1 = block.c1.forward(&x);
            let a1 = relu_vec(&z1);
            let z2 = block.c2.forward(&a1);
            let mut pre = vec![0.0; z2.len()];
            let mut out = vec![0.0; z2.len()];
            for i in 0..z2.len() {
                pre[i] = z2[i] + x[i];
                out[i] = pre[i].max(0.0);
            }
            blocks.push(BlockCache {
                input: x,
                z1,
                a1,
                pre,
            });
            x = out;
        }

        let mut pooled = vec![0.0; self.cfg.channels];
        for c in 0..self.cfg.channels {
            let mut sum = 0.0;
            for idx in 0..GRID_SIZE {
                sum += x[c * GRID_SIZE + idx];
            }
            pooled[c] = sum / GRID_SIZE as f32;
        }

        let mut vh_pre = vec![0.0; self.cfg.value_hidden];
        let mut vh = vec![0.0; self.cfg.value_hidden];
        for h in 0..self.cfg.value_hidden {
            let mut sum = self.value_b1[h];
            for c in 0..self.cfg.channels {
                sum += self.value_w1[h * self.cfg.channels + c] * pooled[c];
            }
            vh_pre[h] = sum;
            vh[h] = sum.max(0.0);
        }
        let mut value_z = self.value_b2;
        for h in 0..self.cfg.value_hidden {
            value_z += self.value_w2[h] * vh[h];
        }
        let value = sigmoid(value_z);

        let mut tile_logits = vec![0.0; GRID_SIZE];
        let mut wildlife_logits = vec![0.0; GRID_SIZE];
        for idx in 0..GRID_SIZE {
            let mut t = self.policy_tile_b;
            let mut w = self.policy_wildlife_b;
            for c in 0..self.cfg.channels {
                let v = x[c * GRID_SIZE + idx];
                t += self.policy_tile_w[c] * v;
                w += self.policy_wildlife_w[c] * v;
            }
            tile_logits[idx] = t;
            wildlife_logits[idx] = w;
        }
        let mut market_logits = [0.0; 4];
        let mut wildlife_market_logits = [0.0; 4];
        for slot in 0..4 {
            let mut m = self.policy_market_b[slot];
            let mut wm = self.policy_wildlife_market_b[slot];
            for c in 0..self.cfg.channels {
                m += self.policy_market_w[slot * self.cfg.channels + c] * pooled[c];
                wm += self.policy_wildlife_market_w[slot * self.cfg.channels + c] * pooled[c];
            }
            market_logits[slot] = m;
            wildlife_market_logits[slot] = wm;
        }
        let mut skip_logit = self.policy_skip_b;
        for c in 0..self.cfg.channels {
            skip_logit += self.policy_skip_w[c] * pooled[c];
        }

        ForwardCache {
            input: input.to_vec(),
            stem_z,
            blocks,
            trunk: x,
            pooled,
            vh_pre,
            vh,
            value,
            tile_logits,
            wildlife_logits,
            market_logits,
            wildlife_market_logits,
            skip_logit,
        }
    }

    pub fn evaluate(&self, game: &GameState, candidates: &[ScoredMove]) -> (f32, Vec<f32>) {
        let input = encode_game(game);
        let cache = self.forward_cache(&input);
        let logits = candidate_logits_from_cache(&cache, candidates);
        (cache.value, softmax(&logits))
    }

    pub fn policy_probs(&self, game: &GameState, candidates: &[ScoredMove]) -> Vec<f32> {
        let input = encode_game(game);
        let cache = self.forward_cache(&input);
        softmax(&candidate_logits_from_cache(&cache, candidates))
    }

    pub fn train_sample(&mut self, sample: &AzSample, lr: f32) -> (f32, f32) {
        if sample.candidates.is_empty() || sample.policy.len() != sample.candidates.len() {
            return (0.0, 0.0);
        }
        let cache = self.forward_cache(&sample.input);
        let logits = candidate_logits_from_cache(&cache, &sample.candidates);
        let probs = softmax(&logits);
        let mut policy_loss = 0.0f32;
        let mut d_logits = vec![0.0; probs.len()];
        for i in 0..probs.len() {
            let target = sample.policy[i].max(0.0);
            policy_loss -= target * probs[i].max(1e-9).ln();
            d_logits[i] = probs[i] - target;
        }

        let value_err = cache.value - sample.value;
        let value_loss = value_err * value_err;
        let mut grad_trunk = vec![0.0; self.cfg.channels * GRID_SIZE];
        self.backward_policy_heads(&cache, &sample.candidates, &d_logits, &mut grad_trunk, lr);
        self.backward_value_head(&cache, sample.value, &mut grad_trunk, lr);
        self.backward_trunk(&cache, grad_trunk, lr);
        (policy_loss, value_loss)
    }

    pub fn train_epochs(&mut self, samples: &[AzSample], epochs: usize, lr: f32) -> TrainStats {
        let mut stats = TrainStats {
            samples: samples.len() * epochs,
            ..TrainStats::default()
        };
        if samples.is_empty() || epochs == 0 {
            return stats;
        }
        for _ in 0..epochs {
            for sample in samples {
                let (pl, vl) = self.train_sample(sample, lr);
                stats.policy_loss += pl;
                stats.value_loss += vl;
            }
        }
        let denom = stats.samples.max(1) as f32;
        stats.policy_loss /= denom;
        stats.value_loss /= denom;
        stats
    }

    fn backward_policy_heads(
        &mut self,
        cache: &ForwardCache,
        candidates: &[ScoredMove],
        d_logits: &[f32],
        grad_trunk: &mut [f32],
        lr: f32,
    ) {
        let channels = self.cfg.channels;
        let mut grad_pooled = vec![0.0; channels];
        let mut gb_tile = 0.0f32;
        let mut gb_wildlife = 0.0f32;
        let mut gb_skip = 0.0f32;
        let mut gw_tile = vec![0.0; channels];
        let mut gw_wildlife = vec![0.0; channels];
        let mut gw_skip = vec![0.0; channels];
        let mut gw_market = vec![0.0; 4 * channels];
        let mut gb_market = [0.0f32; 4];
        let mut gw_wildlife_market = vec![0.0; 4 * channels];
        let mut gb_wildlife_market = [0.0f32; 4];

        for (mv, &d) in candidates.iter().zip(d_logits.iter()) {
            if d == 0.0 {
                continue;
            }
            if let Some(tidx) = move_tile_index(mv) {
                gb_tile += d;
                for c in 0..channels {
                    let v = cache.trunk[c * GRID_SIZE + tidx];
                    gw_tile[c] += d * v;
                    grad_trunk[c * GRID_SIZE + tidx] += d * self.policy_tile_w[c];
                }
            }
            if let Some(widx) = move_wildlife_index(mv) {
                gb_wildlife += d;
                for c in 0..channels {
                    let v = cache.trunk[c * GRID_SIZE + widx];
                    gw_wildlife[c] += d * v;
                    grad_trunk[c * GRID_SIZE + widx] += d * self.policy_wildlife_w[c];
                }
            } else {
                gb_skip += d;
                for c in 0..channels {
                    gw_skip[c] += d * cache.pooled[c];
                    grad_pooled[c] += d * self.policy_skip_w[c];
                }
            }
            if mv.market_index < 4 {
                gb_market[mv.market_index] += d;
                for c in 0..channels {
                    let wi = mv.market_index * channels + c;
                    gw_market[wi] += d * cache.pooled[c];
                    grad_pooled[c] += d * self.policy_market_w[wi];
                }
            }
            let wslot = mv.wildlife_market_index.unwrap_or(mv.market_index);
            if wslot < 4 {
                gb_wildlife_market[wslot] += d;
                for c in 0..channels {
                    let wi = wslot * channels + c;
                    gw_wildlife_market[wi] += d * cache.pooled[c];
                    grad_pooled[c] += d * self.policy_wildlife_market_w[wi];
                }
            }
        }

        for c in 0..channels {
            self.policy_tile_w[c] -= lr * gw_tile[c];
            self.policy_wildlife_w[c] -= lr * gw_wildlife[c];
            self.policy_skip_w[c] -= lr * gw_skip[c];
            let gp = grad_pooled[c] / GRID_SIZE as f32;
            for idx in 0..GRID_SIZE {
                grad_trunk[c * GRID_SIZE + idx] += gp;
            }
        }
        self.policy_tile_b -= lr * gb_tile;
        self.policy_wildlife_b -= lr * gb_wildlife;
        self.policy_skip_b -= lr * gb_skip;
        for slot in 0..4 {
            self.policy_market_b[slot] -= lr * gb_market[slot];
            self.policy_wildlife_market_b[slot] -= lr * gb_wildlife_market[slot];
            for c in 0..channels {
                self.policy_market_w[slot * channels + c] -= lr * gw_market[slot * channels + c];
                self.policy_wildlife_market_w[slot * channels + c] -=
                    lr * gw_wildlife_market[slot * channels + c];
            }
        }
    }

    fn backward_value_head(
        &mut self,
        cache: &ForwardCache,
        target: f32,
        grad_trunk: &mut [f32],
        lr: f32,
    ) {
        let channels = self.cfg.channels;
        let hidden = self.cfg.value_hidden;
        let dz = 2.0 * (cache.value - target) * cache.value * (1.0 - cache.value);
        let mut grad_hidden = vec![0.0; hidden];
        for h in 0..hidden {
            let old = self.value_w2[h];
            grad_hidden[h] += dz * old;
            self.value_w2[h] -= lr * dz * cache.vh[h];
        }
        self.value_b2 -= lr * dz;

        let mut grad_pooled = vec![0.0; channels];
        for h in 0..hidden {
            let dh = if cache.vh_pre[h] > 0.0 {
                grad_hidden[h]
            } else {
                0.0
            };
            self.value_b1[h] -= lr * dh;
            for c in 0..channels {
                let wi = h * channels + c;
                let old = self.value_w1[wi];
                grad_pooled[c] += dh * old;
                self.value_w1[wi] -= lr * dh * cache.pooled[c];
            }
        }
        for c in 0..channels {
            let gp = grad_pooled[c] / GRID_SIZE as f32;
            for idx in 0..GRID_SIZE {
                grad_trunk[c * GRID_SIZE + idx] += gp;
            }
        }
    }

    fn backward_trunk(&mut self, cache: &ForwardCache, mut grad: Vec<f32>, lr: f32) {
        for (bi, block) in self.blocks.iter_mut().enumerate().rev() {
            let bc = &cache.blocks[bi];
            let mut grad_pre = vec![0.0; grad.len()];
            for i in 0..grad.len() {
                grad_pre[i] = if bc.pre[i] > 0.0 { grad[i] } else { 0.0 };
            }
            let mut grad_input = grad_pre.clone();
            let mut grad_a1 = vec![0.0; grad.len()];
            block
                .c2
                .backward_update(&bc.a1, &grad_pre, &mut grad_a1, lr);
            let mut grad_z1 = vec![0.0; grad.len()];
            for i in 0..grad.len() {
                grad_z1[i] = if bc.z1[i] > 0.0 { grad_a1[i] } else { 0.0 };
            }
            block
                .c1
                .backward_update(&bc.input, &grad_z1, &mut grad_input, lr);
            grad = grad_input;
        }
        let mut grad_stem_z = vec![0.0; grad.len()];
        for i in 0..grad.len() {
            grad_stem_z[i] = if cache.stem_z[i] > 0.0 { grad[i] } else { 0.0 };
        }
        let mut _grad_input = vec![0.0; cache.input.len()];
        self.stem
            .backward_update(&cache.input, &grad_stem_z, &mut _grad_input, lr);
    }

    pub fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        use std::io::Write;
        let mut f = std::fs::File::create(path)?;
        f.write_all(AZ_MAGIC)?;
        write_u32(&mut f, self.cfg.channels as u32)?;
        write_u32(&mut f, self.cfg.blocks as u32)?;
        write_u32(&mut f, self.cfg.value_hidden as u32)?;
        write_u32(&mut f, self.cfg.max_candidates as u32)?;
        write_f32(&mut f, self.cfg.c_puct)?;
        self.write_conv(&mut f, &self.stem)?;
        for block in &self.blocks {
            self.write_conv(&mut f, &block.c1)?;
            self.write_conv(&mut f, &block.c2)?;
        }
        write_vec(&mut f, &self.policy_tile_w)?;
        write_f32(&mut f, self.policy_tile_b)?;
        write_vec(&mut f, &self.policy_wildlife_w)?;
        write_f32(&mut f, self.policy_wildlife_b)?;
        write_vec(&mut f, &self.policy_market_w)?;
        for &v in &self.policy_market_b {
            write_f32(&mut f, v)?;
        }
        write_vec(&mut f, &self.policy_wildlife_market_w)?;
        for &v in &self.policy_wildlife_market_b {
            write_f32(&mut f, v)?;
        }
        write_vec(&mut f, &self.policy_skip_w)?;
        write_f32(&mut f, self.policy_skip_b)?;
        write_vec(&mut f, &self.value_w1)?;
        write_vec(&mut f, &self.value_b1)?;
        write_vec(&mut f, &self.value_w2)?;
        write_f32(&mut f, self.value_b2)?;
        Ok(())
    }

    pub fn load(path: &std::path::Path) -> std::io::Result<Self> {
        use std::io::Read;
        let mut f = std::fs::File::open(path)?;
        let mut magic = [0u8; 4];
        f.read_exact(&mut magic)?;
        if &magic != AZ_MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "bad AlphaZero weight magic",
            ));
        }
        let cfg = AlphaZeroConfig {
            channels: read_u32(&mut f)? as usize,
            blocks: read_u32(&mut f)? as usize,
            value_hidden: read_u32(&mut f)? as usize,
            max_candidates: read_u32(&mut f)? as usize,
            c_puct: read_f32(&mut f)?,
        };
        let stem = read_conv(&mut f)?;
        let mut blocks = Vec::with_capacity(cfg.blocks);
        for _ in 0..cfg.blocks {
            blocks.push(ResidualBlock {
                c1: read_conv(&mut f)?,
                c2: read_conv(&mut f)?,
            });
        }
        let policy_tile_w = read_vec(&mut f)?;
        let policy_tile_b = read_f32(&mut f)?;
        let policy_wildlife_w = read_vec(&mut f)?;
        let policy_wildlife_b = read_f32(&mut f)?;
        let policy_market_w = read_vec(&mut f)?;
        let mut policy_market_b = [0.0; 4];
        for v in &mut policy_market_b {
            *v = read_f32(&mut f)?;
        }
        let policy_wildlife_market_w = read_vec(&mut f)?;
        let mut policy_wildlife_market_b = [0.0; 4];
        for v in &mut policy_wildlife_market_b {
            *v = read_f32(&mut f)?;
        }
        let policy_skip_w = read_vec(&mut f)?;
        let policy_skip_b = read_f32(&mut f)?;
        let value_w1 = read_vec(&mut f)?;
        let value_b1 = read_vec(&mut f)?;
        let value_w2 = read_vec(&mut f)?;
        let value_b2 = read_f32(&mut f)?;
        Ok(AlphaZeroNet {
            cfg,
            stem,
            blocks,
            policy_tile_w,
            policy_tile_b,
            policy_wildlife_w,
            policy_wildlife_b,
            policy_market_w,
            policy_market_b,
            policy_wildlife_market_w,
            policy_wildlife_market_b,
            policy_skip_w,
            policy_skip_b,
            value_w1,
            value_b1,
            value_w2,
            value_b2,
        })
    }

    fn write_conv(&self, f: &mut std::fs::File, conv: &Conv2d) -> std::io::Result<()> {
        write_u32(f, conv.in_c as u32)?;
        write_u32(f, conv.out_c as u32)?;
        write_vec(f, &conv.w)?;
        write_vec(f, &conv.b)
    }
}

pub fn save_samples(path: &std::path::Path, samples: &[AzSample]) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;
    f.write_all(AZ_DATA_MAGIC)?;
    write_u32(&mut f, AZ_INPUT_CHANNELS as u32)?;
    write_u32(&mut f, GRID_DIM as u32)?;
    write_u32(&mut f, GRID_SIZE as u32)?;
    write_u32(&mut f, samples.len() as u32)?;
    for sample in samples {
        if sample.input.len() != AZ_INPUT_CHANNELS * GRID_SIZE {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "AlphaZero sample has wrong input length",
            ));
        }
        if sample.policy.len() != sample.candidates.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "AlphaZero sample policy/candidate length mismatch",
            ));
        }
        write_f32(&mut f, sample.value)?;
        write_u32(&mut f, sample.candidates.len() as u32)?;
        for &x in &sample.input {
            write_f32(&mut f, x)?;
        }
        for mv in &sample.candidates {
            write_i32(&mut f, move_tile_index(mv).map(|i| i as i32).unwrap_or(-1))?;
            write_i32(
                &mut f,
                move_wildlife_index(mv).map(|i| i as i32).unwrap_or(-1),
            )?;
            write_i32(&mut f, mv.market_index as i32)?;
            write_i32(
                &mut f,
                mv.wildlife_market_index.map(|i| i as i32).unwrap_or(-1),
            )?;
        }
        for &p in &sample.policy {
            write_f32(&mut f, p)?;
        }
    }
    Ok(())
}

pub fn inspect_samples(path: &std::path::Path) -> std::io::Result<AzDataSummary> {
    use std::io::Read;
    let mut f = std::fs::File::open(path)?;
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic)?;
    if &magic != AZ_DATA_MAGIC {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "bad AlphaZero sample magic",
        ));
    }
    let input_channels = read_u32(&mut f)? as usize;
    let grid_dim = read_u32(&mut f)? as usize;
    let grid_size = read_u32(&mut f)? as usize;
    let samples = read_u32(&mut f)? as usize;
    let input_floats = input_channels * grid_size;
    let mut max_candidates = 0usize;
    let mut buf = [0u8; 4];
    for _ in 0..samples {
        f.read_exact(&mut buf)?; // value
        let n = read_u32(&mut f)? as usize;
        max_candidates = max_candidates.max(n);
        skip_bytes(&mut f, input_floats * 4)?;
        skip_bytes(&mut f, n * 4 * 4)?;
        skip_bytes(&mut f, n * 4)?;
    }
    let mut trailing = [0u8; 1];
    if f.read(&mut trailing)? != 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "trailing AlphaZero sample bytes",
        ));
    }
    Ok(AzDataSummary {
        samples,
        input_channels,
        grid_dim,
        grid_size,
        max_candidates,
    })
}

pub fn encode_game(game: &GameState) -> Vec<f32> {
    let player = game.current_player;
    let board = &game.boards[player];
    let mut input = vec![0.0f32; AZ_INPUT_CHANNELS * GRID_SIZE];
    for idx in 0..GRID_SIZE {
        input[idx] = 1.0; // bias plane
        let cell = board.grid.get(idx);
        if cell.is_present() {
            set_plane(&mut input, 1, idx, 1.0);
            if cell.is_keystone() {
                set_plane(&mut input, 2, idx, 1.0);
            }
            if let Some(t) = cell.primary_terrain() {
                set_plane(&mut input, 3 + t as usize, idx, 1.0);
            }
            if let Some(t) = cell.secondary_terrain() {
                set_plane(&mut input, 3 + t as usize, idx, 1.0);
            }
            if let Some(w) = cell.placed_wildlife() {
                set_plane(&mut input, 8 + w as usize, idx, 1.0);
            }
            for w in Wildlife::ALL {
                if cell.can_place_wildlife(w) {
                    set_plane(&mut input, 13 + w as usize, idx, 1.0);
                }
            }
        }
    }

    let mut market_wildlife = [0.0f32; 5];
    let mut market_terrain = [0.0f32; 5];
    for (_, pair) in game.market.available() {
        market_wildlife[pair.wildlife as usize] += 0.25;
        market_terrain[pair.tile.terrain1 as usize] += 0.25;
        if let Some(t2) = pair.tile.terrain2 {
            market_terrain[t2 as usize] += 0.25;
        }
    }
    for w in 0..5 {
        fill_plane(&mut input, 18 + w, market_wildlife[w]);
    }
    for t in 0..5 {
        fill_plane(&mut input, 23 + t, market_terrain[t].min(1.0));
    }
    fill_plane(&mut input, 28, (board.nature_tokens as f32 / 8.0).min(1.0));
    fill_plane(
        &mut input,
        29,
        (game.turns_remaining as f32 / (20 * game.num_players) as f32).clamp(0.0, 1.0),
    );
    for t in 0..5 {
        fill_plane(
            &mut input,
            30 + t,
            (board.largest_group[t] as f32 / 20.0).min(1.0),
        );
        let opp_max = game
            .boards
            .iter()
            .enumerate()
            .filter(|(p, _)| *p != player)
            .map(|(_, b)| b.largest_group[t])
            .max()
            .unwrap_or(0);
        fill_plane(&mut input, 35 + t, (opp_max as f32 / 20.0).min(1.0));
    }
    for w in 0..5 {
        fill_plane(
            &mut input,
            40 + w,
            (board.wildlife_positions[w].len() as f32 / 20.0).min(1.0),
        );
    }
    let (tile_dist, wildlife_cap) = game.tile_bag.feature_distributions();
    for t in 0..5 {
        fill_plane(&mut input, 45 + t, (tile_dist[t] as f32 / 40.0).min(1.0));
    }
    for w in 0..5 {
        fill_plane(&mut input, 50 + w, (wildlife_cap[w] as f32 / 40.0).min(1.0));
        let opp_max = game
            .boards
            .iter()
            .enumerate()
            .filter(|(p, _)| *p != player)
            .map(|(_, b)| b.wildlife_positions[w].len())
            .max()
            .unwrap_or(0);
        fill_plane(&mut input, 55 + w, (opp_max as f32 / 20.0).min(1.0));
    }
    let opp_token_max = game
        .boards
        .iter()
        .enumerate()
        .filter(|(p, _)| *p != player)
        .map(|(_, b)| b.nature_tokens)
        .max()
        .unwrap_or(0);
    fill_plane(&mut input, 60, (opp_token_max as f32 / 8.0).min(1.0));
    fill_plane(
        &mut input,
        61,
        ((game.num_players.saturating_sub(1)) as f32 / 3.0).min(1.0),
    );
    fill_plane(&mut input, 62, (player as f32 / 3.0).min(1.0));
    fill_plane(
        &mut input,
        63,
        (game.turns_remaining as f32 / 80.0).clamp(0.0, 1.0),
    );
    fill_plane(&mut input, 64, (board.nature_tokens as f32 / 3.0).min(1.0));
    input
}

pub fn candidate_moves(game: &GameState, max_candidates: usize) -> Vec<ScoredMove> {
    let mut candidates = crate::mce::default_greedy_mce_candidates(game);
    let k = max_candidates.max(1);
    if candidates.len() > k {
        candidates.select_nth_unstable_by(k, candidate_rank_cmp);
        candidates.truncate(k);
    }
    candidates.sort_by(candidate_rank_cmp);
    candidates
}

fn candidate_rank_cmp(a: &ScoredMove, b: &ScoredMove) -> Ordering {
    b.eval
        .cmp(&a.eval)
        .then_with(|| a.market_index.cmp(&b.market_index))
        .then_with(|| a.wildlife_market_index.cmp(&b.wildlife_market_index))
        .then_with(|| a.tile_q.cmp(&b.tile_q))
        .then_with(|| a.tile_r.cmp(&b.tile_r))
        .then_with(|| a.rotation.cmp(&b.rotation))
        .then_with(|| a.wildlife_q.cmp(&b.wildlife_q))
        .then_with(|| a.wildlife_r.cmp(&b.wildlife_r))
}

pub fn best_move_alpha_zero(
    game: &GameState,
    net: &AlphaZeroNet,
    simulations: usize,
    rng: &mut StdRng,
) -> Option<ScoredMove> {
    az_search(game, net, simulations, 0.0, rng).map(|r| r.selected)
}

#[derive(Clone)]
pub struct SearchResult {
    pub selected: ScoredMove,
    pub candidates: Vec<ScoredMove>,
    pub visit_policy: Vec<f32>,
    pub visits: Vec<u32>,
}

struct Edge {
    action: ScoredMove,
    prior: f32,
    visits: u32,
    value_sum: f32,
    child: Option<Box<Node>>,
}

struct Node {
    expanded: bool,
    visits: u32,
    edges: Vec<Edge>,
}

impl Node {
    fn new() -> Self {
        Node {
            expanded: false,
            visits: 0,
            edges: Vec::new(),
        }
    }
}

pub fn az_search(
    game: &GameState,
    net: &AlphaZeroNet,
    simulations: usize,
    temperature: f32,
    rng: &mut StdRng,
) -> Option<SearchResult> {
    if should_use_root_parallel(simulations) {
        return az_search_root_parallel(game, net, simulations, temperature, rng);
    }
    az_search_serial(game, net, simulations, temperature, rng)
}

fn az_search_serial(
    game: &GameState,
    net: &AlphaZeroNet,
    simulations: usize,
    temperature: f32,
    rng: &mut StdRng,
) -> Option<SearchResult> {
    let root_player = game.current_player;
    let mut root = Node::new();
    for _ in 0..simulations.max(1) {
        let mut g = game.clone();
        g.shuffle_bags(rng);
        simulate_puct(&mut root, &g, root_player, net, rng);
    }
    if root.edges.is_empty() {
        return None;
    }
    let visits: Vec<u32> = root.edges.iter().map(|e| e.visits).collect();
    let total: u32 = visits.iter().sum();
    let visit_policy: Vec<f32> = if total == 0 {
        vec![1.0 / visits.len() as f32; visits.len()]
    } else {
        visits.iter().map(|&v| v as f32 / total as f32).collect()
    };
    let selected_idx = select_from_visits(&visits, temperature, rng);
    Some(SearchResult {
        selected: root.edges[selected_idx].action,
        candidates: root.edges.iter().map(|e| e.action).collect(),
        visit_policy,
        visits,
    })
}

fn should_use_root_parallel(simulations: usize) -> bool {
    if simulations < 2 {
        return false;
    }
    std::env::var("CASCADIA_AZ_PARALLEL")
        .ok()
        .map(|s| !s.is_empty() && s != "0" && s.to_ascii_lowercase() != "false")
        .unwrap_or(false)
}

fn az_parallel_workers(simulations: usize) -> usize {
    let available = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let requested = std::env::var("CASCADIA_AZ_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(available);
    let min_sims_per_worker = std::env::var("CASCADIA_AZ_MIN_SIMS_PER_THREAD")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4usize)
        .max(1);
    let useful_workers = (simulations / min_sims_per_worker).max(1);
    requested
        .max(1)
        .min(available)
        .min(useful_workers)
        .min(simulations.max(1))
}

fn az_search_root_parallel(
    game: &GameState,
    net: &AlphaZeroNet,
    simulations: usize,
    temperature: f32,
    rng: &mut StdRng,
) -> Option<SearchResult> {
    let workers = az_parallel_workers(simulations);
    if workers <= 1 {
        return az_search_serial(game, net, simulations, temperature, rng);
    }

    let root_candidates = candidate_moves(game, net.cfg.max_candidates);
    if root_candidates.is_empty() {
        return None;
    }
    let base = simulations / workers;
    let rem = simulations % workers;
    let mut seeds = Vec::with_capacity(workers);
    for _ in 0..workers {
        seeds.push(rng.gen::<u64>());
    }

    let mut handles = Vec::with_capacity(workers);
    for (worker, seed) in seeds.into_iter().enumerate() {
        let sims = base + usize::from(worker < rem);
        if sims == 0 {
            continue;
        }
        let g = game.clone();
        let n = net.clone();
        handles.push(std::thread::spawn(move || {
            let mut wrng = StdRng::seed_from_u64(seed);
            az_search_serial(&g, &n, sims, 0.0, &mut wrng)
        }));
    }

    let mut visits = vec![0u32; root_candidates.len()];
    for handle in handles {
        if let Ok(Some(result)) = handle.join() {
            if result.candidates.len() == root_candidates.len()
                && result
                    .candidates
                    .iter()
                    .zip(root_candidates.iter())
                    .all(|(a, b)| same_move(a, b))
            {
                for (dst, src) in visits.iter_mut().zip(result.visits.iter()) {
                    *dst += *src;
                }
            } else {
                for (candidate, src) in result.candidates.iter().zip(result.visits.iter()) {
                    if let Some(idx) = root_candidates
                        .iter()
                        .position(|root| same_move(root, candidate))
                    {
                        visits[idx] += *src;
                    }
                }
            }
        }
    }
    if visits.iter().all(|&v| v == 0) {
        return az_search_serial(game, net, simulations, temperature, rng);
    }

    let total: u32 = visits.iter().sum();
    let visit_policy = visits.iter().map(|&v| v as f32 / total as f32).collect();
    let selected_idx = select_from_visits(&visits, temperature, rng);
    Some(SearchResult {
        selected: root_candidates[selected_idx],
        candidates: root_candidates,
        visit_policy,
        visits,
    })
}

fn simulate_puct(
    node: &mut Node,
    game: &GameState,
    root_player: usize,
    net: &AlphaZeroNet,
    rng: &mut StdRng,
) -> f32 {
    if game.is_game_over() {
        node.visits += 1;
        return score_with_bonus(game, root_player) / AZ_VALUE_SCALE;
    }
    if !node.expanded {
        let candidates = candidate_moves(game, net.cfg.max_candidates);
        let input = encode_game(game);
        let cache = net.forward_cache(&input);
        if candidates.is_empty() {
            node.expanded = true;
            node.visits += 1;
            return cache.value;
        }
        let priors = softmax(&candidate_logits_from_cache(&cache, &candidates));
        node.edges = candidates
            .into_iter()
            .zip(priors.into_iter())
            .map(|(action, prior)| Edge {
                action,
                prior,
                visits: 0,
                value_sum: 0.0,
                child: None,
            })
            .collect();
        node.expanded = true;
        node.visits += 1;
        return cache.value;
    }
    let edge_idx = select_puct(node, net.cfg.c_puct);
    let mut next = game.clone();
    if !execute_scored_move(&mut next, &node.edges[edge_idx].action) {
        node.visits += 1;
        return 0.0;
    }
    advance_to_player_greedy(&mut next, root_player);
    let child = node.edges[edge_idx]
        .child
        .get_or_insert_with(|| Box::new(Node::new()));
    let value = simulate_puct(child, &next, root_player, net, rng);
    let edge = &mut node.edges[edge_idx];
    edge.visits += 1;
    edge.value_sum += value;
    node.visits += 1;
    value
}

fn select_puct(node: &Node, c_puct: f32) -> usize {
    let parent = node.visits.max(1) as f32;
    let mut best = 0usize;
    let mut best_score = f32::NEG_INFINITY;
    for (i, edge) in node.edges.iter().enumerate() {
        let q = if edge.visits == 0 {
            0.5
        } else {
            edge.value_sum / edge.visits as f32
        };
        let u = c_puct * edge.prior * parent.sqrt() / (1.0 + edge.visits as f32);
        let score = q + u;
        if score > best_score {
            best_score = score;
            best = i;
        }
    }
    best
}

fn select_from_visits(visits: &[u32], temperature: f32, rng: &mut StdRng) -> usize {
    if temperature <= 0.01 {
        return visits
            .iter()
            .enumerate()
            .max_by_key(|(_, v)| **v)
            .map(|(i, _)| i)
            .unwrap_or(0);
    }
    let weights: Vec<f32> = visits
        .iter()
        .map(|&v| (v.max(1) as f32).powf(1.0 / temperature))
        .collect();
    let total: f32 = weights.iter().sum();
    let mut r = rng.gen_range(0.0..total.max(1e-9));
    for (i, w) in weights.iter().enumerate() {
        if r <= *w {
            return i;
        }
        r -= *w;
    }
    weights.len().saturating_sub(1)
}

pub fn collect_greedy_bootstrap_games(num_games: usize, rng: &mut StdRng) -> Vec<AzSample> {
    let mut out = Vec::new();
    for _ in 0..num_games {
        let cards = ScoringCards::all_a();
        let mut game = GameState::new(4, cards, rng);
        let mut pending: Vec<(AzSample, usize)> = Vec::new();
        while !game.is_game_over() {
            if game.can_replace_overflow().is_some() {
                game.replace_overflow();
            }
            let player = game.current_player;
            let candidates = candidate_moves(&game, AlphaZeroConfig::default().max_candidates);
            if candidates.is_empty() {
                break;
            }
            let greedy = greedy_move(&game).unwrap_or(candidates[0]);
            let target_idx = candidates
                .iter()
                .position(|mv| same_move(mv, &greedy))
                .unwrap_or(0);
            let mut policy = vec![0.0; candidates.len()];
            policy[target_idx] = 1.0;
            pending.push((
                AzSample {
                    input: encode_game(&game),
                    candidates: candidates.clone(),
                    policy,
                    value: 0.0,
                },
                player,
            ));
            if !execute_scored_move(&mut game, &greedy) {
                break;
            }
        }
        finalize_samples(&game, pending, &mut out);
    }
    out
}

pub fn collect_selfplay_games(
    net: &AlphaZeroNet,
    num_games: usize,
    simulations: usize,
    temperature: f32,
    rng: &mut StdRng,
) -> Vec<AzSample> {
    let mut out = Vec::new();
    for _ in 0..num_games {
        let cards = ScoringCards::all_a();
        let mut game = GameState::new(4, cards, rng);
        let mut pending: Vec<(AzSample, usize)> = Vec::new();
        while !game.is_game_over() {
            if game.can_replace_overflow().is_some() {
                game.replace_overflow();
            }
            let player = game.current_player;
            let input = encode_game(&game);
            let Some(search) = az_search(&game, net, simulations, temperature, rng) else {
                break;
            };
            pending.push((
                AzSample {
                    input,
                    candidates: search.candidates.clone(),
                    policy: search.visit_policy.clone(),
                    value: 0.0,
                },
                player,
            ));
            if !execute_scored_move(&mut game, &search.selected) {
                break;
            }
        }
        finalize_samples(&game, pending, &mut out);
    }
    out
}

pub fn benchmark_alpha_zero(
    net: &AlphaZeroNet,
    games: usize,
    simulations: usize,
    rng: &mut StdRng,
) -> (f32, f32) {
    let mut base = 0.0f32;
    let mut bonus = 0.0f32;
    for _ in 0..games {
        let cards = ScoringCards::all_a();
        let mut game = GameState::new(4, cards, rng);
        while !game.is_game_over() {
            if game.can_replace_overflow().is_some() {
                game.replace_overflow();
            }
            let mv = if game.current_player == 0 {
                best_move_alpha_zero(&game, net, simulations, rng).or_else(|| greedy_move(&game))
            } else {
                greedy_move(&game)
            };
            match mv {
                Some(mv) => {
                    if !execute_scored_move(&mut game, &mv) {
                        break;
                    }
                }
                None => break,
            }
        }
        base +=
            ScoreBreakdown::compute(&mut game.boards[0].clone(), &game.scoring_cards).total as f32;
        bonus += score_with_bonus(&game, 0);
    }
    let denom = games.max(1) as f32;
    (base / denom, bonus / denom)
}

fn finalize_samples(game: &GameState, pending: Vec<(AzSample, usize)>, out: &mut Vec<AzSample>) {
    for (mut sample, player) in pending {
        sample.value = (score_with_bonus(game, player) / AZ_VALUE_SCALE).clamp(0.0, 1.0);
        out.push(sample);
    }
}

fn advance_to_player_greedy(game: &mut GameState, player: usize) {
    while !game.is_game_over() && game.current_player != player {
        if game.can_replace_overflow().is_some() {
            game.replace_overflow();
        }
        match greedy_move(game) {
            Some(mv) => {
                if !execute_scored_move(game, &mv) {
                    break;
                }
            }
            None => break,
        }
    }
}

fn score_with_bonus(game: &GameState, player: usize) -> f32 {
    let mut boards = game.boards.clone();
    ScoreBreakdown::compute_with_bonuses(&mut boards, &game.scoring_cards, player).total as f32
}

fn candidate_logits_from_cache(cache: &ForwardCache, candidates: &[ScoredMove]) -> Vec<f32> {
    candidates
        .iter()
        .map(|mv| {
            let mut logit = 0.0;
            if let Some(tidx) = move_tile_index(mv) {
                logit += cache.tile_logits[tidx];
            } else {
                logit -= 1e9;
            }
            if let Some(widx) = move_wildlife_index(mv) {
                logit += cache.wildlife_logits[widx];
            } else {
                logit += cache.skip_logit;
            }
            if mv.market_index < 4 {
                logit += cache.market_logits[mv.market_index];
            }
            let wslot = mv.wildlife_market_index.unwrap_or(mv.market_index);
            if wslot < 4 {
                logit += cache.wildlife_market_logits[wslot];
            }
            logit
        })
        .collect()
}

fn move_tile_index(mv: &ScoredMove) -> Option<usize> {
    HexCoord::new(mv.tile_q, mv.tile_r).to_index()
}

fn move_wildlife_index(mv: &ScoredMove) -> Option<usize> {
    match (mv.wildlife_q, mv.wildlife_r) {
        (Some(q), Some(r)) => HexCoord::new(q, r).to_index(),
        _ => None,
    }
}

fn same_move(a: &ScoredMove, b: &ScoredMove) -> bool {
    a.market_index == b.market_index
        && a.wildlife_market_index == b.wildlife_market_index
        && a.tile_q == b.tile_q
        && a.tile_r == b.tile_r
        && a.rotation == b.rotation
        && a.wildlife_q == b.wildlife_q
        && a.wildlife_r == b.wildlife_r
}

#[inline]
fn set_plane(input: &mut [f32], plane: usize, idx: usize, value: f32) {
    input[plane * GRID_SIZE + idx] = value;
}

fn fill_plane(input: &mut [f32], plane: usize, value: f32) {
    let start = plane * GRID_SIZE;
    for v in &mut input[start..start + GRID_SIZE] {
        *v = value;
    }
}

fn relu_vec(x: &[f32]) -> Vec<f32> {
    x.iter().map(|v| v.max(0.0)).collect()
}

fn conv_neighbors() -> &'static [Vec<(usize, usize)>] {
    CONV_NEIGHBORS
        .get_or_init(|| {
            let mut all = Vec::with_capacity(GRID_SIZE);
            for q in 0..GRID_DIM {
                for r in 0..GRID_DIM {
                    let mut one = Vec::with_capacity(9);
                    for ky in 0..3 {
                        let nq = q as isize + ky as isize - 1;
                        if !(0..GRID_DIM as isize).contains(&nq) {
                            continue;
                        }
                        for kx in 0..3 {
                            let nr = r as isize + kx as isize - 1;
                            if !(0..GRID_DIM as isize).contains(&nr) {
                                continue;
                            }
                            one.push((ky * 3 + kx, nq as usize * GRID_DIM + nr as usize));
                        }
                    }
                    all.push(one);
                }
            }
            all
        })
        .as_slice()
}

fn rand_vec(rng: &mut StdRng, n: usize, scale: f32) -> Vec<f32> {
    (0..n).map(|_| rng.gen_range(-scale..scale)).collect()
}

fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        let z = (-x).exp();
        1.0 / (1.0 + z)
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|l| (l - max_l).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum <= 0.0 || !sum.is_finite() {
        return vec![1.0 / logits.len() as f32; logits.len()];
    }
    exps.into_iter().map(|e| e / sum).collect()
}

fn write_u32(f: &mut std::fs::File, v: u32) -> std::io::Result<()> {
    use std::io::Write;
    f.write_all(&v.to_le_bytes())
}

fn write_f32(f: &mut std::fs::File, v: f32) -> std::io::Result<()> {
    use std::io::Write;
    f.write_all(&v.to_le_bytes())
}

fn write_i32(f: &mut std::fs::File, v: i32) -> std::io::Result<()> {
    use std::io::Write;
    f.write_all(&v.to_le_bytes())
}

fn write_vec(f: &mut std::fs::File, v: &[f32]) -> std::io::Result<()> {
    write_u32(f, v.len() as u32)?;
    for &x in v {
        write_f32(f, x)?;
    }
    Ok(())
}

fn read_u32(f: &mut std::fs::File) -> std::io::Result<u32> {
    use std::io::Read;
    let mut buf = [0u8; 4];
    f.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_f32(f: &mut std::fs::File) -> std::io::Result<f32> {
    use std::io::Read;
    let mut buf = [0u8; 4];
    f.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

fn skip_bytes(f: &mut std::fs::File, n: usize) -> std::io::Result<()> {
    use std::io::Read;
    let mut remaining = n;
    let mut buf = [0u8; 8192];
    while remaining > 0 {
        let take = remaining.min(buf.len());
        f.read_exact(&mut buf[..take])?;
        remaining -= take;
    }
    Ok(())
}

fn read_vec(f: &mut std::fs::File) -> std::io::Result<Vec<f32>> {
    let len = read_u32(f)? as usize;
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        out.push(read_f32(f)?);
    }
    Ok(out)
}

fn read_conv(f: &mut std::fs::File) -> std::io::Result<Conv2d> {
    let in_c = read_u32(f)? as usize;
    let out_c = read_u32(f)? as usize;
    let w = read_vec(f)?;
    let b = read_vec(f)?;
    Ok(Conv2d { in_c, out_c, w, b })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn encoder_has_expected_shape_and_finite_values() {
        let mut rng = StdRng::seed_from_u64(7);
        let game = GameState::new(4, ScoringCards::all_a(), &mut rng);
        let input = encode_game(&game);
        assert_eq!(input.len(), AZ_INPUT_CHANNELS * GRID_SIZE);
        assert!(input.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn network_policy_is_distribution_over_legal_candidates() {
        let mut rng = StdRng::seed_from_u64(8);
        let game = GameState::new(4, ScoringCards::all_a(), &mut rng);
        let net = AlphaZeroNet::new(AlphaZeroConfig::default(), 9);
        let cands = candidate_moves(&game, 16);
        let (_v, probs) = net.evaluate(&game, &cands);
        assert_eq!(probs.len(), cands.len());
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4);
        assert!(probs.iter().all(|p| p.is_finite() && *p >= 0.0));
    }

    #[test]
    fn save_load_roundtrip_preserves_outputs() {
        let mut rng = StdRng::seed_from_u64(10);
        let game = GameState::new(4, ScoringCards::all_a(), &mut rng);
        let cfg = AlphaZeroConfig {
            channels: 4,
            blocks: 1,
            value_hidden: 8,
            max_candidates: 8,
            c_puct: 2.0,
        };
        let net = AlphaZeroNet::new(cfg, 11);
        let path = std::env::temp_dir().join("cascadia_az_roundtrip.azr");
        net.save(&path).unwrap();
        let loaded = AlphaZeroNet::load(&path).unwrap();
        let cands = candidate_moves(&game, 8);
        let (v1, p1) = net.evaluate(&game, &cands);
        let (v2, p2) = loaded.evaluate(&game, &cands);
        assert!((v1 - v2).abs() < 1e-6);
        for (a, b) in p1.iter().zip(p2.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn puct_returns_legal_move() {
        let mut rng = StdRng::seed_from_u64(12);
        let game = GameState::new(4, ScoringCards::all_a(), &mut rng);
        let cfg = AlphaZeroConfig {
            channels: 4,
            blocks: 1,
            value_hidden: 8,
            max_candidates: 8,
            c_puct: 2.0,
        };
        let net = AlphaZeroNet::new(cfg, 13);
        let result = az_search(&game, &net, 4, 0.0, &mut rng).unwrap();
        assert!(!result.candidates.is_empty());
        assert_eq!(result.visit_policy.len(), result.candidates.len());
        assert!(result
            .candidates
            .iter()
            .any(|m| same_move(m, &result.selected)));
    }

    #[test]
    fn root_parallel_puct_returns_legal_move() {
        let mut rng = StdRng::seed_from_u64(16);
        let game = GameState::new(4, ScoringCards::all_a(), &mut rng);
        let cfg = AlphaZeroConfig {
            channels: 4,
            blocks: 1,
            value_hidden: 8,
            max_candidates: 8,
            c_puct: 2.0,
        };
        let net = AlphaZeroNet::new(cfg, 17);
        let result = az_search_root_parallel(&game, &net, 8, 0.0, &mut rng).unwrap();
        assert!(!result.candidates.is_empty());
        assert_eq!(result.visit_policy.len(), result.candidates.len());
        let total_visits = result.visits.iter().sum::<u32>();
        assert!(total_visits > 0 && total_visits <= 8);
        assert!(result
            .candidates
            .iter()
            .any(|m| same_move(m, &result.selected)));
    }

    #[test]
    fn greedy_bootstrap_labels_are_valid() {
        let mut rng = StdRng::seed_from_u64(14);
        let samples = collect_greedy_bootstrap_games(1, &mut rng);
        assert!(!samples.is_empty());
        for s in samples.iter().take(5) {
            assert_eq!(s.candidates.len(), s.policy.len());
            let sum: f32 = s.policy.iter().sum();
            assert!((sum - 1.0).abs() < 1e-6);
            assert!((0.0..=1.0).contains(&s.value));
        }
    }

    #[test]
    fn sample_file_roundtrip_summary_is_exact() {
        let mut rng = StdRng::seed_from_u64(15);
        let samples = collect_greedy_bootstrap_games(1, &mut rng);
        let path = std::env::temp_dir().join("cascadia_az_samples.azd");
        save_samples(&path, &samples).unwrap();
        let summary = inspect_samples(&path).unwrap();
        assert_eq!(summary.samples, samples.len());
        assert_eq!(summary.input_channels, AZ_INPUT_CHANNELS);
        assert_eq!(summary.grid_dim, GRID_DIM);
        assert_eq!(summary.grid_size, GRID_SIZE);
        assert!(summary.max_candidates > 0);
        let _ = std::fs::remove_file(path);
    }
}
