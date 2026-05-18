//! Standalone policy network for candidate move ranking.
//!
//! Separate from NNUE (which predicts value). This network is trained with
//! cross-entropy ranking loss on MCE score distributions to predict which
//! candidate move MCE would choose. Used for prefilter candidate ranking.
//!
//! Architecture: NUM_FEATURES → H1 → H2 → 1
//! Same sparse binary feature input as NNUE, but wider hidden layers
//! to preserve spatial information needed for ranking similar afterstates.

use crate::nnue::{NUM_FEATURES, extract_features_with_bag, BagInfo, PositionPolicyData};
use cascadia_core::board::Board;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const POLICY_H1: usize = 512;
const POLICY_H2: usize = 256;

pub struct PolicyNetwork {
    pub w1: Vec<f32>,  // [NUM_FEATURES × POLICY_H1]
    pub b1: Vec<f32>,  // [POLICY_H1]
    pub w2: Vec<f32>,  // [POLICY_H1 × POLICY_H2]
    pub b2: Vec<f32>,  // [POLICY_H2]
    pub w3: Vec<f32>,  // [POLICY_H2]
    pub b3: f32,
}

impl Clone for PolicyNetwork {
    fn clone(&self) -> Self {
        PolicyNetwork {
            w1: self.w1.clone(), b1: self.b1.clone(),
            w2: self.w2.clone(), b2: self.b2.clone(),
            w3: self.w3.clone(), b3: self.b3,
        }
    }
}

impl PolicyNetwork {
    pub fn new() -> Self {
        let mut rng = StdRng::seed_from_u64(0xCAFE0042);
        let scale1 = (2.0 / NUM_FEATURES as f32).sqrt();
        let scale2 = (2.0 / POLICY_H1 as f32).sqrt();
        let scale3 = (2.0 / POLICY_H2 as f32).sqrt();

        let w1: Vec<f32> = (0..NUM_FEATURES * POLICY_H1)
            .map(|_| (rng.gen::<f32>() - 0.5) * scale1)
            .collect();
        let b1 = vec![0.0; POLICY_H1];
        let w2: Vec<f32> = (0..POLICY_H1 * POLICY_H2)
            .map(|_| (rng.gen::<f32>() - 0.5) * scale2)
            .collect();
        let b2 = vec![0.0; POLICY_H2];
        let w3: Vec<f32> = (0..POLICY_H2)
            .map(|_| (rng.gen::<f32>() - 0.5) * scale3)
            .collect();

        PolicyNetwork { w1, b1, w2, b2, w3, b3: 0.0 }
    }

    /// Initialize from an existing NNUE value network's weights.
    /// Copies w1/b1 directly (same dimensions). w2 is zero-padded from
    /// NNUE's narrower h2 (64 → 256). w3/b3 are randomly initialized
    /// since they serve a different purpose (ranking vs value).
    pub fn from_nnue(nnue: &crate::nnue::NNUENetwork) -> Self {
        let mut net = Self::new();

        // Copy first layer (same dimensions)
        let copy_len = net.w1.len().min(nnue.w1.len());
        net.w1[..copy_len].copy_from_slice(&nnue.w1[..copy_len]);
        let b1_len = net.b1.len().min(nnue.b1.len());
        net.b1[..b1_len].copy_from_slice(&nnue.b1[..b1_len]);

        // Second layer: copy what fits, rest stays random-initialized.
        // NNUE h2 is HIDDEN2 (64); policy h2 is POLICY_H2 (256).
        // Copy the first HIDDEN2 columns of each row.
        let nnue_h2 = crate::nnue::HIDDEN2;
        for i in 0..POLICY_H1.min(crate::nnue::HIDDEN1) {
            let src_base = i * nnue_h2;
            let dst_base = i * POLICY_H2;
            let cols = nnue_h2.min(POLICY_H2);
            if src_base + cols <= nnue.w2.len() && dst_base + cols <= net.w2.len() {
                net.w2[dst_base..dst_base + cols].copy_from_slice(&nnue.w2[src_base..src_base + cols]);
            }
        }
        let b2_copy = nnue_h2.min(POLICY_H2).min(nnue.b2.len());
        net.b2[..b2_copy].copy_from_slice(&nnue.b2[..b2_copy]);

        // w3/b3: fresh random (ranking head, not value head)
        net
    }

    pub fn forward(&self, features: &[u16]) -> f32 {
        let mut h1 = [0.0f32; POLICY_H1];
        h1.copy_from_slice(&self.b1);
        for &fi in features {
            let base = fi as usize * POLICY_H1;
            if base + POLICY_H1 > self.w1.len() { continue; }
            let col = &self.w1[base..base + POLICY_H1];
            for j in 0..POLICY_H1 { h1[j] += col[j]; }
        }
        for v in h1.iter_mut() { *v = v.max(0.0); }

        let mut h2 = vec![0.0f32; POLICY_H2];
        for j in 0..POLICY_H2 { h2[j] = self.b2[j]; }
        for i in 0..POLICY_H1 {
            if h1[i] > 0.0 {
                let base = i * POLICY_H2;
                for j in 0..POLICY_H2 { h2[j] += h1[i] * self.w2[base + j]; }
            }
        }
        for v in h2.iter_mut() { *v = v.max(0.0); }

        let mut out = self.b3;
        for j in 0..POLICY_H2 { out += h2[j] * self.w3[j]; }
        out
    }

    /// Train on one position (N candidates) with cross-entropy ranking loss.
    /// Full backprop through all layers.
    pub fn train_ranking(&mut self, pos: &PositionPolicyData, lr: f32, temperature: f32) -> (f32, bool) {
        let n = pos.candidates.len();
        if n < 2 { return (0.0, false); }

        // Forward all candidates
        let mut h1s: Vec<[f32; POLICY_H1]> = Vec::with_capacity(n);
        let mut h1_pres: Vec<[f32; POLICY_H1]> = Vec::with_capacity(n);
        let mut h2s: Vec<Vec<f32>> = Vec::with_capacity(n);
        let mut h2_pres: Vec<Vec<f32>> = Vec::with_capacity(n);
        let mut logits: Vec<f32> = Vec::with_capacity(n);

        for (feats, _) in &pos.candidates {
            let mut h1 = [0.0f32; POLICY_H1];
            h1.copy_from_slice(&self.b1);
            for &fi in feats {
                let base = fi as usize * POLICY_H1;
                if base + POLICY_H1 > self.w1.len() { continue; }
                let col = &self.w1[base..base + POLICY_H1];
                for j in 0..POLICY_H1 { h1[j] += col[j]; }
            }
            let mut h1_pre = [0.0f32; POLICY_H1];
            h1_pre.copy_from_slice(&h1);
            for v in h1.iter_mut() { *v = v.max(0.0); }

            let mut h2 = vec![0.0f32; POLICY_H2];
            for j in 0..POLICY_H2 { h2[j] = self.b2[j]; }
            for i in 0..POLICY_H1 {
                if h1[i] > 0.0 {
                    let base = i * POLICY_H2;
                    for j in 0..POLICY_H2 { h2[j] += h1[i] * self.w2[base + j]; }
                }
            }
            let mut h2_pre = h2.clone();
            for v in h2.iter_mut() { *v = v.max(0.0); }

            let mut out = self.b3;
            for j in 0..POLICY_H2 { out += h2[j] * self.w3[j]; }
            logits.push(out);

            h1s.push(h1); h1_pres.push(h1_pre);
            h2s.push(h2); h2_pres.push(h2_pre);
        }

        // Target + prediction
        let mce_scores: Vec<f32> = pos.candidates.iter().map(|(_, s)| *s).collect();
        let max_mce = mce_scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let target: Vec<f32> = {
            let exps: Vec<f32> = mce_scores.iter().map(|s| ((s - max_mce) / temperature).exp()).collect();
            let sum: f32 = exps.iter().sum();
            exps.iter().map(|e| e / sum.max(1e-8)).collect()
        };
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let pred: Vec<f32> = {
            let exps: Vec<f32> = logits.iter().map(|l| (l - max_logit).exp()).collect();
            let sum: f32 = exps.iter().sum();
            exps.iter().map(|e| e / sum.max(1e-8)).collect()
        };

        let loss: f32 = target.iter().zip(pred.iter())
            .map(|(t, p)| -t * p.max(1e-8).ln()).sum();
        let mce_best = mce_scores.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap_or(0);
        let pred_best = logits.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap_or(0);

        // Backprop
        let scaled_lr = lr / n as f32;
        for ci in 0..n {
            let d_out = (pred[ci] - target[ci]) * scaled_lr;

            for j in 0..POLICY_H2 { self.w3[j] -= d_out * h2s[ci][j]; }
            self.b3 -= d_out;

            let mut d_h2 = vec![0.0f32; POLICY_H2];
            for j in 0..POLICY_H2 {
                if h2_pres[ci][j] > 0.0 { d_h2[j] = d_out * self.w3[j]; }
            }

            for i in 0..POLICY_H1 {
                if h1s[ci][i] > 0.0 {
                    let base = i * POLICY_H2;
                    for j in 0..POLICY_H2 { self.w2[base + j] -= d_h2[j] * h1s[ci][i]; }
                }
            }
            for j in 0..POLICY_H2 { self.b2[j] -= d_h2[j]; }

            let mut d_h1 = [0.0f32; POLICY_H1];
            for i in 0..POLICY_H1 {
                if h1_pres[ci][i] > 0.0 {
                    let base = i * POLICY_H2;
                    for j in 0..POLICY_H2 { d_h1[i] += d_h2[j] * self.w2[base + j]; }
                }
            }

            let feats = &pos.candidates[ci].0;
            for &fi in feats {
                let base = fi as usize * POLICY_H1;
                if base + POLICY_H1 > self.w1.len() { continue; }
                for j in 0..POLICY_H1 { self.w1[base + j] -= d_h1[j]; }
            }
            for j in 0..POLICY_H1 { self.b1[j] -= d_h1[j]; }
        }

        (loss, mce_best == pred_best)
    }

    pub fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        use std::io::Write;
        let mut f = std::fs::File::create(path)?;
        f.write_all(b"PLCY")?;
        let version: u32 = 1;
        f.write_all(&version.to_le_bytes())?;
        let h1: u32 = POLICY_H1 as u32;
        let h2: u32 = POLICY_H2 as u32;
        let nf: u32 = NUM_FEATURES as u32;
        f.write_all(&nf.to_le_bytes())?;
        f.write_all(&h1.to_le_bytes())?;
        f.write_all(&h2.to_le_bytes())?;
        for &v in &self.w1 { f.write_all(&v.to_le_bytes())?; }
        for &v in &self.b1 { f.write_all(&v.to_le_bytes())?; }
        for &v in &self.w2 { f.write_all(&v.to_le_bytes())?; }
        for &v in &self.b2 { f.write_all(&v.to_le_bytes())?; }
        for &v in &self.w3 { f.write_all(&v.to_le_bytes())?; }
        f.write_all(&self.b3.to_le_bytes())?;
        Ok(())
    }

    pub fn load(path: &std::path::Path) -> std::io::Result<Self> {
        use std::io::Read;
        let mut f = std::fs::File::open(path)?;
        let mut magic = [0u8; 4];
        f.read_exact(&mut magic)?;
        if &magic != b"PLCY" {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "bad policy magic"));
        }
        let mut buf4 = [0u8; 4];
        f.read_exact(&mut buf4)?; // version
        f.read_exact(&mut buf4)?;
        let nf = u32::from_le_bytes(buf4) as usize;
        f.read_exact(&mut buf4)?;
        let h1 = u32::from_le_bytes(buf4) as usize;
        f.read_exact(&mut buf4)?;
        let h2 = u32::from_le_bytes(buf4) as usize;

        let mut read_f32 = |file: &mut std::fs::File| -> std::io::Result<f32> {
            let mut b = [0u8; 4];
            file.read_exact(&mut b)?;
            Ok(f32::from_le_bytes(b))
        };

        let mut w1 = Vec::with_capacity(nf * h1);
        for _ in 0..nf * h1 { w1.push(read_f32(&mut f)?); }
        let mut b1 = Vec::with_capacity(h1);
        for _ in 0..h1 { b1.push(read_f32(&mut f)?); }
        let mut w2 = Vec::with_capacity(h1 * h2);
        for _ in 0..h1 * h2 { w2.push(read_f32(&mut f)?); }
        let mut b2 = Vec::with_capacity(h2);
        for _ in 0..h2 { b2.push(read_f32(&mut f)?); }
        let mut w3 = Vec::with_capacity(h2);
        for _ in 0..h2 { w3.push(read_f32(&mut f)?); }
        let b3 = read_f32(&mut f)?;

        Ok(PolicyNetwork { w1, b1, w2, b2, w3, b3 })
    }
}
