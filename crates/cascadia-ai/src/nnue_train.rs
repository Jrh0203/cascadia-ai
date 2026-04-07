//! NNUE training: generate self-play data, train network with mini-batch SGD.

use std::sync::Arc;
use std::thread;

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

use cascadia_core::board::Board;
use cascadia_core::game::GameState;
use cascadia_core::hex::HexCoord;
use cascadia_core::scoring::ScoreBreakdown;
use cascadia_core::types::ScoringCards;

use crate::nnue::{extract_features, extract_phase_pattern_features, NNUENetwork};
use crate::search::{execute_scored_move, greedy_move};

/// A training sample: board features + target score.
#[derive(Clone)]
pub struct Sample {
    pub features: Vec<u16>,
    pub target: f32,
}

/// Generate training data from self-play games.
/// `num_players`: 1 for pre-training (AI gets all turns), 4 for realistic play.
fn generate_samples(num_games: usize, seed: u64, net: Option<&NNUENetwork>, epsilon: f32, num_players: usize) -> Vec<Sample> {
    let num_threads = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let games_per_thread = (num_games + num_threads - 1) / num_threads;

    let handles: Vec<_> = (0..num_threads)
        .map(|t| {
            let n = if t < num_threads - 1 {
                games_per_thread.min(num_games.saturating_sub(t * games_per_thread))
            } else {
                num_games.saturating_sub(t * games_per_thread)
            };
            let thread_seed = seed.wrapping_add(t as u64 * 1000000);
            let net_clone = net.cloned();
            let epsilon = epsilon;
            let num_players = num_players;
            thread::spawn(move || {
                let mut rng = StdRng::seed_from_u64(thread_seed);
                let mut samples = Vec::with_capacity(n * 20);

                for _ in 0..n {
                    let game_seed = rng.gen::<u64>();
                    generate_game_samples(&mut samples, game_seed, net_clone.as_ref(), epsilon, num_players);
                }

                samples
            })
        })
        .collect();

    let mut all_samples = Vec::with_capacity(num_games * 20);
    for handle in handles {
        all_samples.extend(handle.join().unwrap());
    }
    all_samples
}

/// Play one game, record all AI afterstates, label with final score.
fn generate_game_samples(samples: &mut Vec<Sample>, seed: u64, net: Option<&NNUENetwork>, epsilon: f32, num_players: usize) {
    let mut rng = StdRng::seed_from_u64(seed);
    let cards = ScoringCards::all_a();
    let mut game = GameState::new(num_players, cards, &mut rng);

    // Collect afterstate features + scores during the game
    let mut afterstates: Vec<(Vec<u16>, u16)> = Vec::with_capacity(20);

    while !game.is_game_over() {
        if game.current_player != 0 {
            // Opponents play greedy (must match benchmark for consistency)
            match greedy_move(&game) {
                Some(mv) => {
                    if !execute_scored_move(&mut game, &mv) { break; }
                }
                None => break,
            }
            continue;
        }

        // Pre-move: simple greedy mulligan logic for training data generation
        greedy_pre_move(&mut game, &mut rng);

        // Epsilon-greedy: with probability epsilon, pick a random valid move
        let mv = if epsilon > 0.0 && rng.gen::<f32>() < epsilon {
            pick_random_move(&game, &mut rng)
        } else {
            match net {
                Some(n) => pick_best_move_nnue(&game, n),
                None => greedy_move(&game),
            }
        };
        match mv {
            Some(mv) => {
                if !execute_scored_move(&mut game, &mv) { break; }
            }
            None => break,
        }

        // Record afterstate features + current score
        let current_score = ScoreBreakdown::compute(
            &mut game.boards[0], &game.scoring_cards,
        ).total;
        let bag_info = crate::nnue::BagInfo::from_game(&game);
        afterstates.push((crate::nnue::extract_features_with_bag(&game.boards[0], Some(&bag_info)), current_score));
    }

    // Final score
    let final_score = ScoreBreakdown::compute(
        &mut game.boards[0], &game.scoring_cards,
    ).total;

    // Delta labels: remaining points to gain
    for (features, current_score) in &afterstates {
        let remaining = final_score.saturating_sub(*current_score) as f32;
        samples.push(Sample { features: features.clone(), target: remaining });
    }

    // Cache high-scoring games (use global mutex for thread-safe writes)
    if final_score >= 90 {
        use std::sync::{Mutex, OnceLock};
        static CACHE_MUTEX: OnceLock<Mutex<()>> = OnceLock::new();
        let mutex = CACHE_MUTEX.get_or_init(|| Mutex::new(()));
        let _guard = mutex.lock().unwrap();

        let cache_path = std::path::Path::new("training_cache_90plus.bin");
        if let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true).append(true).open(cache_path)
        {
            use std::io::Write;
            // Build the full record in memory first, then write atomically
            let mut buf: Vec<u8> = Vec::with_capacity(1024);
            buf.extend_from_slice(&(afterstates.len() as u16).to_le_bytes());
            buf.extend_from_slice(&final_score.to_le_bytes());
            for (features, current_score) in &afterstates {
                buf.extend_from_slice(&(features.len() as u16).to_le_bytes());
                for &f in features {
                    buf.extend_from_slice(&f.to_le_bytes());
                }
                buf.extend_from_slice(&current_score.to_le_bytes());
            }
            let _ = file.write_all(&buf);
        }
    }
}

/// Load samples from the high-score game cache.
/// Each game in the cache scored 90+ during training. Labels use the delta
/// scheme: target = final_score - current_score (remaining points to gain).
pub fn load_cache_samples(cache_path: &std::path::Path) -> std::io::Result<Vec<Sample>> {
    use std::io::Read;
    let mut file = std::fs::File::open(cache_path)?;
    let mut samples = Vec::new();
    let mut buf2 = [0u8; 2];

    // Read all bytes first — easier to handle truncated files
    let mut all_bytes = Vec::new();
    file.read_to_end(&mut all_bytes)?;
    let mut pos = 0usize;

    let mut read_u16 = |bytes: &[u8], pos: &mut usize| -> Option<u16> {
        if *pos + 2 > bytes.len() { return None; }
        let v = u16::from_le_bytes([bytes[*pos], bytes[*pos + 1]]);
        *pos += 2;
        Some(v)
    };

    let mut games_loaded = 0;
    let mut games_skipped = 0;

    while pos < all_bytes.len() {
        let game_start = pos;
        // Read num_positions
        let n = match read_u16(&all_bytes, &mut pos) {
            Some(v) => v as usize,
            None => break,
        };
        let final_score = match read_u16(&all_bytes, &mut pos) {
            Some(v) => v,
            None => break,
        };

        // Validate: num_positions should be reasonable (0..25)
        if n > 25 {
            // Looks corrupted at this offset
            pos = game_start + 2;
            games_skipped += 1;
            if games_skipped > 100 { break; }
            continue;
        }

        let mut game_samples: Vec<Sample> = Vec::with_capacity(n);
        let mut game_ok = true;

        for _ in 0..n {
            let nf = match read_u16(&all_bytes, &mut pos) {
                Some(v) => v as usize,
                None => { game_ok = false; break; }
            };
            if nf > 300 { game_ok = false; break; }

            let mut features = Vec::with_capacity(nf);
            for _ in 0..nf {
                match read_u16(&all_bytes, &mut pos) {
                    Some(v) => features.push(v),
                    None => { game_ok = false; break; }
                }
            }
            if !game_ok { break; }

            let current_score = match read_u16(&all_bytes, &mut pos) {
                Some(v) => v,
                None => { game_ok = false; break; }
            };
            let target = final_score.saturating_sub(current_score) as f32;
            game_samples.push(Sample { features, target });
        }

        if game_ok {
            samples.extend(game_samples);
            games_loaded += 1;
        } else {
            // Try to resync: advance 2 bytes and retry
            pos = game_start + 2;
            games_skipped += 1;
            if games_skipped > 100 { break; } // too corrupted
        }
    }

    eprintln!("  [loaded {} games, skipped {} corrupted]", games_loaded, games_skipped);
    let _ = buf2; // silence unused
    Ok(samples)
}

// ── MCE Policy Samples: flat file format ──
// Magic: 4 bytes b"MCEP"
// For each sample:
//   u16 nf, nf × u16 features, f32 target
const MCE_POLICY_MAGIC: &[u8; 4] = b"MCEP";

/// Append MCE-labeled samples to a file. Creates the file with a magic header if new.
pub fn append_mce_samples(
    path: &std::path::Path,
    samples: &[(Vec<u16>, f32)],
) -> std::io::Result<()> {
    use std::io::Write;
    let is_new = !path.exists();
    let mut file = std::fs::OpenOptions::new()
        .create(true).append(true).open(path)?;
    let mut buf: Vec<u8> = Vec::with_capacity(samples.len() * 64);
    if is_new {
        buf.extend_from_slice(MCE_POLICY_MAGIC);
    }
    for (features, target) in samples {
        buf.extend_from_slice(&(features.len() as u16).to_le_bytes());
        for &f in features {
            buf.extend_from_slice(&f.to_le_bytes());
        }
        buf.extend_from_slice(&target.to_le_bytes());
    }
    file.write_all(&buf)?;
    Ok(())
}

/// Load all MCE policy samples from a file.
pub fn load_mce_samples(path: &std::path::Path) -> std::io::Result<Vec<Sample>> {
    use std::io::Read;
    let mut file = std::fs::File::open(path)?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    let mut pos = 0usize;
    if bytes.len() < 4 || &bytes[..4] != MCE_POLICY_MAGIC {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "bad magic"));
    }
    pos += 4;
    let mut samples = Vec::new();
    while pos + 2 <= bytes.len() {
        let nf = u16::from_le_bytes([bytes[pos], bytes[pos+1]]) as usize;
        pos += 2;
        if nf > 1024 || pos + nf * 2 + 4 > bytes.len() { break; }
        let mut features = Vec::with_capacity(nf);
        for _ in 0..nf {
            features.push(u16::from_le_bytes([bytes[pos], bytes[pos+1]]));
            pos += 2;
        }
        let target = f32::from_le_bytes([bytes[pos], bytes[pos+1], bytes[pos+2], bytes[pos+3]]);
        pos += 4;
        samples.push(Sample { features, target });
    }
    Ok(samples)
}

// ── Hex rotation augmentation ──
// 120° CW in axial coords: (q, r) → (-q-r, q)
// 240° CW in axial coords: (q, r) → (r, -q-r)
// Pairwise line directions cycle: 0→1→2→0 under 120° CW.

const GRID_DIM: usize = 21;
const GRID_CENTER: i8 = 10;

/// Build a cell index rotation table. Returns None entries for cells that rotate out of bounds.
fn build_rotation_table(rot: usize) -> [Option<usize>; 441] {
    let mut table = [None; 441];
    for idx in 0..441 {
        let q = (idx / GRID_DIM) as i8 - GRID_CENTER;
        let r = (idx % GRID_DIM) as i8 - GRID_CENTER;
        let (q2, r2) = match rot {
            1 => (-q - r, q),       // 120° CW
            2 => (r, -q - r),       // 240° CW
            _ => (q, r),
        };
        let col = q2 as i16 + GRID_CENTER as i16;
        let row = r2 as i16 + GRID_CENTER as i16;
        if col >= 0 && col < GRID_DIM as i16 && row >= 0 && row < GRID_DIM as i16 {
            table[idx] = Some(col as usize * GRID_DIM + row as usize);
        }
    }
    table
}

/// Whether a pairwise pair should swap (my, neighbor) order when rotating.
/// Under 120° CW: E→SW(reverse), NE→SE(reverse), NW→E(forward).
/// Under 240° CW: E→NW(forward), NE→W(reverse), NW→SW(reverse).
/// Swap when the rotated direction is a REVERSE direction (raw dir >= 3).
const PAIR_SWAP: [[bool; 3]; 3] = [
    [false, false, false], // dir_shift=0: identity, no swap
    [true,  true,  false], // dir_shift=1 (120° CW): dirs 0,1 swap; dir 2 doesn't
    [false, true,  true],  // dir_shift=2 (240° CW): dirs 1,2 swap; dir 0 doesn't
];

/// Swap a wildlife pairwise pair_state: (my*7 + n) → (n*7 + my)
#[inline]
fn swap_wl_pair(pair_state: usize) -> usize {
    let my = pair_state / 7;
    let n = pair_state % 7;
    n * 7 + my
}

/// Swap a terrain pairwise pair_state: (my*6 + n) → (n*6 + my)
#[inline]
fn swap_terrain_pair(pair_state: usize) -> usize {
    let my = pair_state / 6;
    let n = pair_state % 6;
    n * 6 + my
}

/// Rotate a sparse feature vector. Returns None if any active cell rotates out of bounds.
fn rotate_features(features: &[u16], rotation_table: &[Option<usize>; 441], dir_shift: usize) -> Option<Vec<u16>> {
    use crate::nnue;

    // Feature block boundaries (must match nnue.rs layout)
    const FPC: usize = 11; // FEATURES_PER_CELL
    const CELL_END: usize = 441 * FPC; // 4851
    const PHASE_END: usize = CELL_END + 110; // 4961
    const WL_PAIR_STATES: usize = 49;
    const WL_PAIR_END: usize = PHASE_END + 3 * WL_PAIR_STATES; // 5108
    const PATTERN_END: usize = WL_PAIR_END + 89; // 5197
    const BAG_END: usize = PATTERN_END + 55; // 5252
    const OPP_HAB_END: usize = BAG_END + 55; // 5307
    // Allowed wildlife: 441 cells × 5 flags
    const ALLOWED_WL_PC: usize = 5;
    const ALLOWED_END: usize = OPP_HAB_END + 441 * ALLOWED_WL_PC; // 7512
    const EXT_WL_END: usize = ALLOWED_END + 50; // 7562
    // Terrain pairwise: 3 dirs × 36 states
    const TERRAIN_PAIR_STATES: usize = 36;
    const TERRAIN_PAIR_END: usize = EXT_WL_END + 3 * TERRAIN_PAIR_STATES; // 7670

    let mut rotated = Vec::with_capacity(features.len());
    for &f in features {
        let fi = f as usize;
        if fi < CELL_END {
            let cell_idx = fi / FPC;
            let offset = fi % FPC;
            let new_cell = rotation_table[cell_idx]?;
            rotated.push((new_cell * FPC + offset) as u16);
        } else if fi < PHASE_END {
            rotated.push(f);
        } else if fi < WL_PAIR_END {
            let rel = fi - PHASE_END;
            let dir = rel / WL_PAIR_STATES;
            let mut pair_state = rel % WL_PAIR_STATES;
            let new_dir = (dir + dir_shift) % 3;
            if PAIR_SWAP[dir_shift][dir] {
                pair_state = swap_wl_pair(pair_state);
            }
            rotated.push((PHASE_END + new_dir * WL_PAIR_STATES + pair_state) as u16);
        } else if fi < PATTERN_END {
            rotated.push(f);
        } else if fi < OPP_HAB_END {
            rotated.push(f);
        } else if fi < ALLOWED_END {
            let rel = fi - OPP_HAB_END;
            let cell_idx = rel / ALLOWED_WL_PC;
            let offset = rel % ALLOWED_WL_PC;
            let new_cell = rotation_table[cell_idx]?;
            rotated.push((OPP_HAB_END + new_cell * ALLOWED_WL_PC + offset) as u16);
        } else if fi < EXT_WL_END {
            rotated.push(f);
        } else if fi < TERRAIN_PAIR_END {
            let rel = fi - EXT_WL_END;
            let dir = rel / TERRAIN_PAIR_STATES;
            let mut pair_state = rel % TERRAIN_PAIR_STATES;
            let new_dir = (dir + dir_shift) % 3;
            if PAIR_SWAP[dir_shift][dir] {
                pair_state = swap_terrain_pair(pair_state);
            }
            rotated.push((EXT_WL_END + new_dir * TERRAIN_PAIR_STATES + pair_state) as u16);
        } else {
            rotated.push(f);
        }
    }
    Some(rotated)
}

/// Build a cell index translation table for shifting by (dq, dr).
fn build_translation_table(dq: i8, dr: i8) -> [Option<usize>; 441] {
    let mut table = [None; 441];
    for idx in 0..441 {
        let q = (idx / GRID_DIM) as i8 - GRID_CENTER;
        let r = (idx % GRID_DIM) as i8 - GRID_CENTER;
        let q2 = q + dq;
        let r2 = r + dr;
        let col = q2 as i16 + GRID_CENTER as i16;
        let row = r2 as i16 + GRID_CENTER as i16;
        if col >= 0 && col < GRID_DIM as i16 && row >= 0 && row < GRID_DIM as i16 {
            table[idx] = Some(col as usize * GRID_DIM + row as usize);
        }
    }
    table
}

/// Translate a sparse feature vector (shift all cell indices, no direction change).
fn translate_features(features: &[u16], table: &[Option<usize>; 441]) -> Option<Vec<u16>> {
    // Translation is rotation with dir_shift=0 (no direction change)
    rotate_features(features, table, 0)
}

/// Augment samples with rotations (3×) and translations (up to 25×).
/// Combined: up to 75× data augmentation.
/// Public wrapper for augmentation (used by --export-pytorch)
pub fn augment_samples_pub(samples: &[Sample]) -> Vec<Sample> {
    augment_with_rotations(samples)
}

fn augment_with_rotations(samples: &[Sample]) -> Vec<Sample> {
    let table_120 = build_rotation_table(1);
    let table_240 = build_rotation_table(2);

    // Translation offsets: ±2 in q and r = 5×5 = 25 offsets (including (0,0) = identity)
    let mut translation_tables: Vec<(i8, i8, [Option<usize>; 441])> = Vec::new();
    for dq in -2i8..=2 {
        for dr in -2i8..=2 {
            if dq == 0 && dr == 0 { continue; } // skip identity
            translation_tables.push((dq, dr, build_translation_table(dq, dr)));
        }
    }

    // Total: 1 original + 2 rotations + 24 translations + 48 (translations × 2 rotations)
    let max_factor = 1 + 2 + 24 + 48; // 75
    let mut augmented = Vec::with_capacity(samples.len() * max_factor);
    let mut skipped = 0usize;

    for sample in samples {
        // Original
        augmented.push(sample.clone());

        // 2 rotations of original
        if let Some(rot) = rotate_features(&sample.features, &table_120, 1) {
            augmented.push(Sample { features: rot, target: sample.target });
        } else { skipped += 1; }
        if let Some(rot) = rotate_features(&sample.features, &table_240, 2) {
            augmented.push(Sample { features: rot, target: sample.target });
        } else { skipped += 1; }

        // 24 translations
        for &(dq, dr, ref table) in &translation_tables {
            if let Some(trans) = translate_features(&sample.features, table) {
                // 2 rotations of each translation
                if let Some(rot) = rotate_features(&trans, &table_120, 1) {
                    augmented.push(Sample { features: rot, target: sample.target });
                } else { skipped += 1; }
                if let Some(rot) = rotate_features(&trans, &table_240, 2) {
                    augmented.push(Sample { features: rot, target: sample.target });
                } else { skipped += 1; }

                // The translation itself (after rotations so we still have `trans`)
                augmented.push(Sample { features: trans, target: sample.target });
            } else { skipped += 1; }
        }
    }

    if skipped > 0 {
        eprintln!("  [augmentation: skipped {} out-of-bounds transforms]", skipped);
    }
    augmented
}

/// Train NNUE from MCE policy samples (imitation of MCE via regression on rollout averages).
/// If checkpoint_path is provided, saves weights after every epoch.
pub fn train_from_mce_samples(
    net: &mut NNUENetwork,
    samples_path: &std::path::Path,
    epochs: usize,
    lr: f32,
) -> std::io::Result<TrainStats> {
    train_from_mce_samples_with_checkpoint(net, samples_path, epochs, lr, None, 0)
}

pub fn train_from_mce_samples_with_checkpoint(
    net: &mut NNUENetwork,
    samples_path: &std::path::Path,
    epochs: usize,
    lr: f32,
    checkpoint_path: Option<&std::path::Path>,
    freeze_below: usize, // 0 = train all, >0 = only train features >= this index
) -> std::io::Result<TrainStats> {
    let mut stats = TrainStats::default();
    eprint!("  Loading MCE samples from {:?}...", samples_path);
    let start = std::time::Instant::now();
    let raw_samples = load_mce_samples(samples_path)?;
    eprintln!(" {} samples in {:.1?}", raw_samples.len(), start.elapsed());
    if raw_samples.is_empty() {
        return Ok(stats);
    }

    // Augment with 120° and 240° hex rotations (3× data)
    eprint!("  Augmenting with hex rotations...");
    let aug_start = std::time::Instant::now();
    let mut samples = augment_with_rotations(&raw_samples);
    eprintln!(" {} → {} samples in {:.1?}", raw_samples.len(), samples.len(), aug_start.elapsed());
    stats.num_samples = samples.len();

    let mut rng = StdRng::seed_from_u64(42);
    let batch_size = 256;

    let num_threads: usize = std::env::var("CASCADIA_TRAIN_THREADS")
        .ok().and_then(|s| s.parse().ok())
        .unwrap_or(1);

    // Learning rate schedule: warmup for first 3 epochs, then cosine decay
    let warmup_epochs = 3.min(epochs);
    let lr_schedule = |epoch: usize| -> f32 {
        if epoch < warmup_epochs {
            // Linear warmup: 0.1*lr → lr
            let t = (epoch + 1) as f32 / warmup_epochs as f32;
            lr * (0.1 + 0.9 * t)
        } else {
            // Cosine decay: lr → 0.01*lr
            let t = (epoch - warmup_epochs) as f32 / (epochs - warmup_epochs).max(1) as f32;
            let cosine = 0.5 * (1.0 + (std::f32::consts::PI * t).cos());
            lr * (0.01 + 0.99 * cosine)
        }
    };

    for epoch in 0..epochs {
        let epoch_lr = lr_schedule(epoch);
        samples.shuffle(&mut rng);

        let (loss, count) = if num_threads > 1 {
            // Parallel training: split samples across threads, each trains
            // a local copy, then average weights back.
            let chunk_size = (samples.len() + num_threads - 1) / num_threads;
            let net_arc = std::sync::Arc::new(net.clone());
            let samples_arc = std::sync::Arc::new(samples.clone());

            let handles: Vec<_> = (0..num_threads).map(|t| {
                let net_copy = (*net_arc).clone();
                let samples_ref = std::sync::Arc::clone(&samples_arc);
                let start = t * chunk_size;
                let end = ((t + 1) * chunk_size).min(samples_ref.len());
                let lr = epoch_lr;
                let freeze_below = freeze_below;
                let batch_size = batch_size;

                thread::spawn(move || {
                    let mut local_net = net_copy;
                    let mut loss = 0.0f64;
                    let mut count = 0usize;
                    for batch_start in (start..end).step_by(batch_size) {
                        let batch_end = (batch_start + batch_size).min(end);
                        let batch_lr = lr / (batch_end - batch_start) as f32;
                        for sample in &samples_ref[batch_start..batch_end] {
                            let l = if freeze_below > 0 {
                                local_net.train_sample_frozen(&sample.features, sample.target, batch_lr, freeze_below)
                            } else {
                                local_net.train_sample(&sample.features, sample.target, batch_lr)
                            };
                            loss += l as f64;
                            count += 1;
                        }
                    }
                    (local_net, loss, count)
                })
            }).collect();

            let mut total_loss = 0.0f64;
            let mut total_count = 0usize;
            let mut trained_nets: Vec<NNUENetwork> = Vec::with_capacity(num_threads);
            for handle in handles {
                let (local_net, loss, count) = handle.join().unwrap();
                total_loss += loss;
                total_count += count;
                trained_nets.push(local_net);
            }

            // Average all thread-local networks back into master
            net.average_from(&trained_nets);

            (total_loss, total_count)
        } else {
            // Single-threaded (original path)
            let mut loss = 0.0f64;
            let mut count = 0usize;
            for batch_start in (0..samples.len()).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(samples.len());
                let batch_lr = epoch_lr / (batch_end - batch_start) as f32;
                for sample in &samples[batch_start..batch_end] {
                    let l = if freeze_below > 0 {
                        net.train_sample_frozen(&sample.features, sample.target, batch_lr, freeze_below)
                    } else {
                        net.train_sample(&sample.features, sample.target, batch_lr)
                    };
                    loss += l as f64;
                    count += 1;
                }
            }
            (loss, count)
        };

        let rmse = (loss / count as f64).sqrt();
        let thread_str = if num_threads > 1 { format!(" [{}T]", num_threads) } else { String::new() };
        eprint!("\r  Epoch {}/{}: RMSE={:.2} lr={:.6}{}{}    ", epoch + 1, epochs, rmse, epoch_lr,
            if freeze_below > 0 { format!(" [frozen<{}]", freeze_below) } else { String::new() },
            thread_str);
        stats.final_rmse = rmse;

        // Save checkpoint after every epoch
        if let Some(path) = checkpoint_path {
            let _ = net.save(path);
        }
    }
    eprintln!();
    Ok(stats)
}

/// Train NNUE from the high-score cache file (expert imitation learning).
/// This trains on ~1000+ games that scored 90+, labeled with delta targets.
pub fn train_from_cache(
    net: &mut NNUENetwork,
    cache_path: &std::path::Path,
    epochs: usize,
    lr: f32,
) -> std::io::Result<TrainStats> {
    let mut stats = TrainStats::default();
    eprint!("  Loading cache from {:?}...", cache_path);
    let start = std::time::Instant::now();
    let mut samples = load_cache_samples(cache_path)?;
    eprintln!(" {} samples in {:.1?}", samples.len(), start.elapsed());
    stats.num_samples = samples.len();

    let mut rng = StdRng::seed_from_u64(42);
    let batch_size = 256;

    for epoch in 0..epochs {
        samples.shuffle(&mut rng);
        let mut loss = 0.0f64;
        let mut count = 0usize;
        for batch_start in (0..samples.len()).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(samples.len());
            let batch_lr = lr / (batch_end - batch_start) as f32;
            for sample in &samples[batch_start..batch_end] {
                let l = net.train_sample(&sample.features, sample.target, batch_lr);
                loss += l as f64;
                count += 1;
            }
        }
        let rmse = (loss / count as f64).sqrt();
        eprint!("\r  Epoch {}/{}: RMSE={:.2}    ", epoch + 1, epochs, rmse);
        stats.final_rmse = rmse;
    }
    eprintln!();

    Ok(stats)
}

/// Train the NNUE network with optional self-play iterations.
/// iterations=1: train on greedy data only (default).
/// iterations>1: first iteration uses greedy, subsequent use NNUE-guided self-play.
pub fn train_nnue(
    net: &mut NNUENetwork,
    num_games: usize,
    epochs: usize,
    lr: f32,
    seed: u64,
) -> TrainStats {
    let iterations: usize = std::env::args()
        .position(|a| a == "--iterations")
        .and_then(|i| std::env::args().nth(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);

    let epsilon: f32 = std::env::args()
        .position(|a| a == "--epsilon")
        .and_then(|i| std::env::args().nth(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);

    let pretrain: bool = std::env::args().any(|a| a == "--pretrain");

    let mut stats = TrainStats::default();
    let mut rng = StdRng::seed_from_u64(seed + 999);

    // Phase 0: 1-player pre-training (if --pretrain)
    // Teaches the network what high-scoring boards look like
    if pretrain {
        let pretrain_iters = 3;
        for iter in 0..pretrain_iters {
            let use_net = if iter == 0 { None } else { Some(&*net) };
            let iter_epsilon = if iter == 0 { 0.0 } else { epsilon.max(0.05) };
            let label = if use_net.is_some() { "1p self-play" } else { "1p greedy" };

            eprint!("  Pre-train {}/{} ({}): generating {} games...",
                iter + 1, pretrain_iters, label, num_games);
            let start = std::time::Instant::now();
            let mut samples = generate_samples(num_games, seed + iter as u64 * 99999, use_net, iter_epsilon, 1);
            let gen_time = start.elapsed();
            eprintln!(" {} samples in {:.1?}", samples.len(), gen_time);

            let batch_size = 256;
            for epoch in 0..epochs {
                samples.shuffle(&mut rng);
                let mut epoch_loss = 0.0f64;
                let mut epoch_count = 0usize;
                for batch_start in (0..samples.len()).step_by(batch_size) {
                    let batch_end = (batch_start + batch_size).min(samples.len());
                    let batch_lr = lr / (batch_end - batch_start) as f32;
                    for sample in &samples[batch_start..batch_end] {
                        let loss = net.train_sample(&sample.features, sample.target, batch_lr);
                        epoch_loss += loss as f64;
                        epoch_count += 1;
                    }
                }
                let rmse = (epoch_loss / epoch_count as f64).sqrt();
                eprint!("\r  Pre {}, Epoch {}/{}: RMSE={:.2}    ", iter + 1, epoch + 1, epochs, rmse);
                stats.final_rmse = rmse;
            }
            eprintln!();
        }
        eprintln!("  Pre-training complete. Fine-tuning on 4p...");
    }

    // Main training: 4-player iterations
    for iter in 0..iterations {
        let use_net = if iter == 0 && !pretrain { None } else { Some(&*net) };
        let iter_epsilon = if iter == 0 && !pretrain { 0.0 } else { epsilon };
        let iter_label = if use_net.is_some() {
            if iter_epsilon > 0.0 { "4p self-play+explore" } else { "4p self-play" }
        } else { "4p greedy" };

        eprint!("  Iteration {}/{} ({}): generating {} games...",
            iter + 1, iterations, iter_label, num_games);
        let start = std::time::Instant::now();
        let mut samples = generate_samples(num_games, seed + iter as u64 * 12345, use_net, iter_epsilon, 4);
        let gen_time = start.elapsed();
        eprintln!(" {} samples in {:.1?}", samples.len(), gen_time);

        stats.num_samples = samples.len();
        let batch_size = 256;

        for epoch in 0..epochs {
            samples.shuffle(&mut rng);

            let mut epoch_loss = 0.0f64;
            let mut epoch_count = 0usize;

            for batch_start in (0..samples.len()).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(samples.len());
                let batch_lr = lr / (batch_end - batch_start) as f32;

                for sample in &samples[batch_start..batch_end] {
                    let loss = net.train_sample(&sample.features, sample.target, batch_lr);
                    epoch_loss += loss as f64;
                    epoch_count += 1;
                }
            }

            let avg_loss = epoch_loss / epoch_count as f64;
            let rmse = avg_loss.sqrt();
            eprint!("\r  Iter {}, Epoch {}/{}: RMSE={:.2}    ", iter + 1, epoch + 1, epochs, rmse);
            stats.final_rmse = rmse;
        }
        // Save weights after each iteration
        let weights_path = std::env::args()
            .position(|a| a == "--weights")
            .and_then(|i| std::env::args().nth(i + 1))
            .unwrap_or_else(|| "nnue_weights.bin".to_string());
        let _ = net.save(std::path::Path::new(&weights_path));
        eprintln!("  [saved to {}]", weights_path);
    }

    stats
}

/// Compute the marginal value of each AI-placed tile in the final board.
/// For each tile: how much would the score drop if this tile weren't there?
/// Wildlife: analytically compute per-token contribution to pattern scores.
/// Habitat: each tile contributes 1 per terrain (simplified, no group splitting).
/// Returns marginals in placement order (index 0 = first tile placed).
fn compute_tile_marginals(board: &Board, cards: &ScoringCards) -> Vec<f32> {
    let adj = &*cascadia_core::hex::ADJACENCY;
    let mut marginals = Vec::with_capacity(board.placed_tiles.len());

    // Pre-compute pattern info for wildlife marginals
    let bear_pairs = count_bear_pairs_list(board, adj);
    let bear_pair_count = bear_pairs.len();
    let elk_lines = compute_elk_line_lengths(board);
    let salmon_runs = compute_salmon_run_lengths(board, adj);
    let hawk_isolated = count_isolated_hawks(board, adj);

    // Skip first 3 tiles (starter tiles, not AI-placed)
    let ai_start = 3.min(board.placed_tiles.len());

    for i in 0..board.placed_tiles.len() {
        if i < ai_start {
            // Starter tiles — not counted
            continue;
        }
        let idx = board.placed_tiles[i] as usize;
        let cell = board.grid.get(idx);
        let mut marginal = 0.0f32;

        // Habitat marginal: 1 per terrain on this tile
        if cell.primary_terrain().is_some() { marginal += 1.0; }
        if cell.secondary_terrain().is_some() { marginal += 1.0; }

        // Wildlife marginal
        if let Some(w) = cell.placed_wildlife() {
            let variant = cards.variant_for(w);
            marginal += wildlife_marginal(board, idx, w, variant, adj,
                bear_pair_count, &elk_lines, &salmon_runs, hawk_isolated);

            // Nature token from keystone
            if cell.is_keystone() { marginal += 1.0; }
        }

        marginals.push(marginal);
    }

    marginals
}

/// Compute marginal value of a specific wildlife token at `pos`.
fn wildlife_marginal(
    board: &Board, pos: usize, w: cascadia_core::types::Wildlife,
    _variant: cascadia_core::types::ScoringCardVariant,
    adj: &cascadia_core::hex::AdjacencyTable,
    bear_pair_count: usize,
    elk_lines: &[(usize, usize)], // (position, line_length)
    salmon_runs: &[(usize, usize)], // (position, run_length)
    hawk_isolated: usize,
) -> f32 {
    use cascadia_core::types::Wildlife;

    match w {
        Wildlife::Bear => {
            // Check if this bear is part of a valid pair
            let bear_neighbors: usize = adj.neighbors_of(pos)
                .filter(|&n| board.grid.get(n).placed_wildlife() == Some(Wildlife::Bear))
                .count();
            if bear_neighbors == 1 {
                // Part of a pair — marginal = half the pair's marginal value
                // Going from N pairs to N-1: score table [0,4,11,19,27]
                let pair_scores = [0.0, 4.0, 11.0, 19.0, 27.0];
                let n = bear_pair_count.min(4);
                let with = pair_scores[n];
                let without = if n > 0 { pair_scores[n - 1] } else { 0.0 };
                (with - without) / 2.0 // split credit between both bears
            } else {
                0.0 // isolated or in cluster — no scoring contribution
            }
        }
        Wildlife::Elk => {
            // Find the line this elk belongs to
            if let Some(&(_, line_len)) = elk_lines.iter().find(|&&(p, _)| p == pos) {
                let line_scores = [0.0, 2.0, 5.0, 9.0, 13.0];
                let len = line_len.min(4);
                let with = line_scores[len];
                let without = if len > 0 { line_scores[len - 1] } else { 0.0 };
                with - without // marginal of this elk extending the line by 1
            } else {
                2.0 // single elk = 2 points
            }
        }
        Wildlife::Salmon => {
            // Find the run this salmon belongs to
            if let Some(&(_, run_len)) = salmon_runs.iter().find(|&&(p, _)| p == pos) {
                let run_scores = [0.0, 2.0, 4.0, 7.0, 11.0, 15.0, 20.0, 26.0];
                let len = run_len.min(7);
                let with = run_scores[len];
                let without = if len > 0 { run_scores[len - 1] } else { 0.0 };
                with - without
            } else {
                2.0
            }
        }
        Wildlife::Hawk => {
            let has_hawk_neighbor = adj.neighbors_of(pos)
                .any(|n| board.grid.get(n).placed_wildlife() == Some(Wildlife::Hawk));
            if !has_hawk_neighbor {
                // Isolated — marginal of Nth isolated hawk
                let hawk_scores = [0.0, 2.0, 5.0, 8.0, 11.0, 14.0, 18.0, 22.0, 28.0];
                let n = hawk_isolated.min(8);
                let with = hawk_scores[n];
                let without = if n > 0 { hawk_scores[n - 1] } else { 0.0 };
                with - without
            } else {
                0.0
            }
        }
        Wildlife::Fox => {
            // Individual fox score = unique adjacent wildlife types
            let mut mask = 0u8;
            for nidx in adj.neighbors_of(pos) {
                if let Some(w) = board.grid.get(nidx).placed_wildlife() {
                    mask |= 1 << (w as u8);
                }
            }
            mask.count_ones() as f32
        }
    }
}

// Helper: list all positions that are part of bear pairs
fn count_bear_pairs_list(board: &Board, adj: &cascadia_core::hex::AdjacencyTable) -> Vec<(usize, usize)> {
    use cascadia_core::types::Wildlife;
    let positions = &board.wildlife_positions[Wildlife::Bear as usize];
    let mut visited = [false; 441];
    let mut pairs = Vec::new();
    for &pos in positions.iter() {
        let idx = pos as usize;
        if visited[idx] { continue; }
        let mut component = arrayvec::ArrayVec::<u16, 24>::new();
        let mut queue = arrayvec::ArrayVec::<u16, 24>::new();
        queue.push(pos);
        visited[idx] = true;
        while let Some(current) = queue.pop() {
            component.push(current);
            for nidx in adj.neighbors_of(current as usize) {
                if !visited[nidx] && board.grid.get(nidx).placed_wildlife() == Some(Wildlife::Bear) {
                    visited[nidx] = true;
                    queue.push(nidx as u16);
                }
            }
        }
        if component.len() == 2 {
            pairs.push((component[0] as usize, component[1] as usize));
        }
    }
    pairs
}

// Helper: for each elk, find the line it belongs to and the line length
fn compute_elk_line_lengths(board: &Board) -> Vec<(usize, usize)> {
    use cascadia_core::types::Wildlife;
    let positions = &board.wildlife_positions[Wildlife::Elk as usize];
    let mut results = Vec::new();
    let mut is_elk = [false; 441];
    for &pos in positions.iter() { is_elk[pos as usize] = true; }

    // For each elk, find the longest line through it in any direction
    for &pos in positions.iter() {
        let coord = HexCoord::from_index(pos as usize);
        let mut best_len = 1;
        for &(dq, dr) in &HexCoord::LINE_DIRECTIONS {
            let mut len = 1;
            let mut c = HexCoord::new(coord.q + dq, coord.r + dr);
            while let Some(idx) = c.to_index() {
                if is_elk[idx] { len += 1; c = HexCoord::new(c.q + dq, c.r + dr); }
                else { break; }
            }
            c = HexCoord::new(coord.q - dq, coord.r - dr);
            while let Some(idx) = c.to_index() {
                if is_elk[idx] { len += 1; c = HexCoord::new(c.q - dq, c.r - dr); }
                else { break; }
            }
            best_len = best_len.max(len);
        }
        results.push((pos as usize, best_len));
    }
    results
}

// Helper: for each salmon, find the run it belongs to and run length
fn compute_salmon_run_lengths(board: &Board, adj: &cascadia_core::hex::AdjacencyTable) -> Vec<(usize, usize)> {
    use cascadia_core::types::Wildlife;
    let positions = &board.wildlife_positions[Wildlife::Salmon as usize];
    let mut visited = [false; 441];
    let mut results = Vec::new();

    for &pos in positions.iter() {
        let idx = pos as usize;
        if visited[idx] { continue; }
        let mut component = arrayvec::ArrayVec::<u16, 24>::new();
        let mut queue = arrayvec::ArrayVec::<u16, 24>::new();
        queue.push(pos);
        visited[idx] = true;
        while let Some(current) = queue.pop() {
            component.push(current);
            for nidx in adj.neighbors_of(current as usize) {
                if !visited[nidx] && board.grid.get(nidx).placed_wildlife() == Some(Wildlife::Salmon) {
                    visited[nidx] = true;
                    queue.push(nidx as u16);
                }
            }
        }
        let is_valid = component.iter().all(|&p| {
            adj.neighbors_of(p as usize)
                .filter(|&n| board.grid.get(n).placed_wildlife() == Some(Wildlife::Salmon))
                .count() <= 2
        });
        let len = if is_valid { component.len() } else { 0 };
        for &p in &component {
            results.push((p as usize, len));
        }
    }
    results
}

// Helper: count isolated hawks
fn count_isolated_hawks(board: &Board, adj: &cascadia_core::hex::AdjacencyTable) -> usize {
    use cascadia_core::types::Wildlife;
    board.wildlife_positions[Wildlife::Hawk as usize].iter()
        .filter(|&&pos| {
            !adj.neighbors_of(pos as usize)
                .any(|n| board.grid.get(n).placed_wildlife() == Some(Wildlife::Hawk))
        })
        .count()
}

/// Simple pre-move optimization for training data generation.
/// Checks whether replacing 3-of-a-kind or mulliganing improves the greedy score.
fn greedy_pre_move(game: &mut GameState, _rng: &mut StdRng) {
    const MAX_MULLIGANS: usize = 3;
    let player = game.current_player;
    let mut mulligans_used = 0;

    loop {
        let baseline = greedy_score(game);

        // Option 1: free 3-of-a-kind replacement
        if game.can_replace_overflow().is_some() {
            let mut test = game.clone();
            test.replace_overflow();
            if greedy_score(&test) > baseline {
                game.replace_overflow();
                continue;
            }
        }

        // Option 2: paid mulligan (only if significantly better, to offset token cost)
        if mulligans_used < MAX_MULLIGANS && game.boards[player].nature_tokens > 0 {
            let mut test = game.clone();
            if test.mulligan_wildlife() {
                // Use greedy eval on actual post-mulligan state (no sampling for speed)
                let new_score = greedy_score(&test);
                if new_score > baseline + 2 {
                    game.mulligan_wildlife();
                    mulligans_used += 1;
                    continue;
                }
            }
        }
        break;
    }
}

fn greedy_score(game: &GameState) -> u16 {
    greedy_move(game).map(|m| m.score).unwrap_or(0)
}

/// Pick a random valid move (for epsilon-greedy exploration).
fn pick_random_move(game: &GameState, rng: &mut StdRng) -> Option<crate::eval::ScoredMove> {
    use cascadia_core::hex::HexCoord;
    use crate::eval::ScoredMove;

    let mp: Vec<_> = game.market.available()
        .map(|(i, p)| (i, p.tile, p.wildlife)).collect();
    if mp.is_empty() { return None; }

    let board = &game.boards[game.current_player];
    let frontier = board.frontier();
    if frontier.is_empty() { return None; }

    // Pick random market pair
    let &(idx, tile, wildlife) = &mp[rng.gen_range(0..mp.len())];

    // Pick random frontier cell
    let fi = frontier[rng.gen_range(0..frontier.len())] as usize;
    let coord = HexCoord::from_index(fi);

    // Pick random rotation
    let max_rot: u8 = if tile.terrain2.is_none() { 1 } else { 6 };
    let rot = rng.gen_range(0..max_rot);

    // Try to place tile; if invalid, fall back to greedy
    let mut board_clone = board.clone();
    if board_clone.place_tile(coord, tile, rot).is_none() {
        return greedy_move(game);
    }

    // Pick random wildlife placement (or skip with 20% chance)
    let valid_positions: Vec<u16> = board_clone.placed_tiles.iter()
        .copied()
        .filter(|&ti| board_clone.grid.get(ti as usize).can_place_wildlife(wildlife))
        .collect();

    let (wq, wr) = if !valid_positions.is_empty() && rng.gen::<f32>() > 0.2 {
        let ti = valid_positions[rng.gen_range(0..valid_positions.len())];
        let wc = HexCoord::from_index(ti as usize);
        (Some(wc.q), Some(wc.r))
    } else {
        (None, None)
    };

    Some(ScoredMove {
        market_index: idx,
        tile_q: coord.q,
        tile_r: coord.r,
        rotation: rot,
        wildlife_q: wq,
        wildlife_r: wr,
        score: 0,
        eval: 0,
        wildlife_market_index: None,
    })
}

/// Pick best move: get greedy top-K candidates, re-rank by NNUE afterstate value.
pub fn pick_best_move_nnue(
    game: &GameState,
    net: &NNUENetwork,
) -> Option<crate::eval::ScoredMove> {
    use crate::eval::ScoredMove;
    use cascadia_core::hex::HexCoord;

    let mp: Vec<_> = game.market.available()
        .map(|(i, p)| (i, p.tile, p.wildlife)).collect();
    if mp.is_empty() { return None; }

    let cards = game.scoring_cards;
    let turns = game.turns_remaining;
    let player = game.current_player;
    let mut board = game.boards[player].clone();
    let base_move = crate::eval::best_move_with_potential(&mut board, &mp, &cards, turns);

    let mut candidates: Vec<ScoredMove> = crate::search::candidate_moves_pub(game);
    if let Some(ref bm) = base_move {
        if !candidates.iter().any(|c| c.tile_q == bm.tile_q && c.tile_r == bm.tile_r
            && c.rotation == bm.rotation && c.wildlife_q == bm.wildlife_q && c.wildlife_r == bm.wildlife_r) {
            candidates.push(*bm);
        }
    }
    candidates.truncate(15);

    if candidates.is_empty() {
        return base_move;
    }

    // Compute BagInfo once (reused across all candidates)
    let bag_info = crate::nnue::BagInfo::from_game_for_player(game, player);

    // Re-rank by actual_score + NNUE(remaining_value) = estimated final score.
    // Clone only the current player's board (not full GameState) for each candidate.
    let mut best: Option<(ScoredMove, f32)> = None;
    for mv in &candidates {
        let coord = HexCoord::new(mv.tile_q, mv.tile_r);
        let tile = match mp.iter().find(|&&(i, _, _)| i == mv.market_index) {
            Some(&(_, tile, _)) => tile,
            None => continue,
        };
        let wildlife = match mp.iter().find(|&&(i, _, _)| {
            i == mv.wildlife_market_index.unwrap_or(mv.market_index)
        }) {
            Some(&(_, _, wl)) => wl,
            None => continue,
        };

        // Clone just the board (~15KB vs ~60KB+ for full GameState)
        let mut eval_board = board.clone();

        // Place tile
        if eval_board.place_tile(coord, tile, mv.rotation).is_none() {
            continue;
        }

        // Place wildlife
        if let (Some(wq), Some(wr)) = (mv.wildlife_q, mv.wildlife_r) {
            let wcoord = HexCoord::new(wq, wr);
            if let Some(idx) = wcoord.to_index() {
                eval_board.place_wildlife(idx, wildlife);
            }
        }

        // Nature token adjustment for independent draft
        if mv.wildlife_market_index.is_some() {
            eval_board.nature_tokens = eval_board.nature_tokens.saturating_sub(1);
        }

        let actual = cascadia_core::scoring::ScoreBreakdown::compute(
            &mut eval_board, &cards,
        ).total as f32;
        let remaining = net.evaluate_with_bag(&eval_board, &bag_info);
        let estimated_final = actual + remaining;

        if best.is_none() || estimated_final > best.as_ref().unwrap().1 {
            best = Some((*mv, estimated_final));
        }
    }

    best.map(|(mv, _)| mv)
}

/// Enumerate ALL legal moves and score each afterstate with NNUE.
/// No pre-filtering — every (market, frontier, rotation, wildlife_placement) combo is evaluated.
pub fn pick_best_move_nnue_full(
    game: &GameState,
    net: &NNUENetwork,
) -> Option<crate::eval::ScoredMove> {
    use crate::eval::ScoredMove;
    use cascadia_core::scoring::ScoreBreakdown;

    let player = game.current_player;
    let board = &game.boards[player];
    let cards = game.scoring_cards;
    let frontier = board.frontier();
    if frontier.is_empty() { return None; }

    let market_pairs: Vec<_> = game.market.available()
        .map(|(i, p)| (i, p.tile, p.wildlife)).collect();
    if market_pairs.is_empty() { return None; }

    let mut board_clone = board.clone();
    let mut best: Option<(ScoredMove, f32)> = None;

    for &(mi, tile, wildlife) in &market_pairs {
        let max_rot: u8 = if tile.terrain2.is_none() { 1 } else { 6 };

        for &fi in frontier.iter() {
            let coord = HexCoord::from_index(fi as usize);
            for rot in 0..max_rot {
                let tile_action = match board_clone.place_tile(coord, tile, rot) {
                    Some(a) => a,
                    None => continue,
                };

                // Option 1: skip wildlife placement
                let actual = ScoreBreakdown::compute(&mut board_clone, &cards).total as f32;
                let remaining = net.evaluate(&board_clone);
                let score_skip = actual + remaining;

                let skip_mv = ScoredMove {
                    market_index: mi,
                    tile_q: coord.q,
                    tile_r: coord.r,
                    rotation: rot,
                    wildlife_q: None,
                    wildlife_r: None,
                    score: actual as u16,
                    eval: 0,
                    wildlife_market_index: None,
                };
                if best.is_none() || score_skip > best.as_ref().unwrap().1 {
                    best = Some((skip_mv, score_skip));
                }

                // Option 2: try every valid wildlife placement
                let placed: arrayvec::ArrayVec<u16, 64> =
                    board_clone.placed_tiles.iter().copied().collect();
                for &ti in placed.iter() {
                    if !board_clone.grid.get(ti as usize).can_place_wildlife(wildlife) {
                        continue;
                    }
                    let wl_action = match board_clone.place_wildlife(ti as usize, wildlife) {
                        Some(a) => a,
                        None => continue,
                    };

                    let actual_w = ScoreBreakdown::compute(&mut board_clone, &cards).total as f32;
                    let remaining_w = net.evaluate(&board_clone);
                    let score_w = actual_w + remaining_w;

                    board_clone.undo(wl_action);

                    if score_w > best.as_ref().map(|b| b.1).unwrap_or(f32::NEG_INFINITY) {
                        let wc = HexCoord::from_index(ti as usize);
                        best = Some((ScoredMove {
                            market_index: mi,
                            tile_q: coord.q,
                            tile_r: coord.r,
                            rotation: rot,
                            wildlife_q: Some(wc.q),
                            wildlife_r: Some(wc.r),
                            score: actual_w as u16,
                            eval: 0,
                            wildlife_market_index: None,
                        }, score_w));
                    }
                }

                board_clone.undo(tile_action);
            }
        }
    }

    best.map(|(mv, _)| mv)
}

#[derive(Default)]
pub struct TrainStats {
    pub num_samples: usize,
    pub final_rmse: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_swap_wl_pair() {
        // bear(1) looking at salmon(3) → pair_state = 1*7+3 = 10
        // swapped: salmon(3) looking at bear(1) → 3*7+1 = 22
        assert_eq!(swap_wl_pair(10), 22);
        assert_eq!(swap_wl_pair(22), 10);
        // identity: bear(1) looking at bear(1) → 1*7+1 = 8
        assert_eq!(swap_wl_pair(8), 8);
        // empty(0) looking at hawk(4) → 0*7+4 = 4, swapped → 4*7+0 = 28
        assert_eq!(swap_wl_pair(4), 28);
        assert_eq!(swap_wl_pair(28), 4);
    }

    #[test]
    fn test_swap_terrain_pair() {
        // forest(1) next to river(5) → 1*6+5 = 11
        // swapped: river(5) next to forest(1) → 5*6+1 = 31
        assert_eq!(swap_terrain_pair(11), 31);
        assert_eq!(swap_terrain_pair(31), 11);
        // same terrain: mountain(4) next to mountain(4) → 4*6+4 = 28
        assert_eq!(swap_terrain_pair(28), 28);
    }

    #[test]
    fn test_pair_swap_table_consistency() {
        // dir_shift=0 should never swap (identity)
        for dir in 0..3 {
            assert!(!PAIR_SWAP[0][dir]);
        }
        // Rotation 1 (120° CW): dirs 0,1 swap, dir 2 doesn't
        assert!(PAIR_SWAP[1][0]);
        assert!(PAIR_SWAP[1][1]);
        assert!(!PAIR_SWAP[1][2]);
        // Rotation 2 (240° CW): dirs 1,2 swap, dir 0 doesn't
        assert!(!PAIR_SWAP[2][0]);
        assert!(PAIR_SWAP[2][1]);
        assert!(PAIR_SWAP[2][2]);
    }

    #[test]
    fn test_rotate_pairwise_feature_swap() {
        // Create a feature: bear(1) looking at salmon(3) in direction E (dir 0)
        // pair_state = 1*7+3 = 10, feature = PHASE_END + 0*49 + 10 = 4961 + 10 = 4971
        let feature = 4971u16;

        let table_120 = build_rotation_table(1);

        // Rotate 120° CW: dir 0 → dir 1, and pair should SWAP
        // Swapped pair_state = 3*7+1 = 22
        // Expected: PHASE_END + 1*49 + 22 = 4961 + 49 + 22 = 5032
        let rotated = rotate_features(&[feature], &table_120, 1).unwrap();
        assert_eq!(rotated[0], 5032);

        // Without swap it would be 4961 + 49 + 10 = 5020 (wrong)
        assert_ne!(rotated[0], 5020);
    }

    #[test]
    fn test_rotate_pairwise_no_swap_when_forward() {
        // Direction 2 (NW) with rotation 1 → dir 0. NW→E is forward, NO swap.
        // bear(1) looking at elk(2) in dir NW: pair_state = 1*7+2 = 9
        // feature = PHASE_END + 2*49 + 9 = 4961 + 98 + 9 = 5068
        let feature = 5068u16;

        let table_120 = build_rotation_table(1);
        let rotated = rotate_features(&[feature], &table_120, 1).unwrap();

        // dir 2 → dir 0, pair_state stays 9 (no swap)
        // Expected: PHASE_END + 0*49 + 9 = 4961 + 9 = 4970
        assert_eq!(rotated[0], 4970);
    }

    #[test]
    fn test_rotation_120_then_240_is_identity_for_pairwise() {
        // Rotating 120° then 240° should give back the original feature
        let table_120 = build_rotation_table(1);
        let table_240 = build_rotation_table(2);

        // Test several pairwise features
        for dir in 0..3 {
            for my in 0..7 {
                for n in 0..7 {
                    if my == 0 && n == 0 { continue; }
                    let pair_state = my * 7 + n;
                    let fi = (4961 + dir * 49 + pair_state) as u16;
                    let rot1 = rotate_features(&[fi], &table_120, 1).unwrap();
                    let rot2 = rotate_features(&rot1, &table_240, 2).unwrap();
                    assert_eq!(rot2[0], fi,
                        "120+240 not identity for dir={}, my={}, n={}: {} → {} → {}",
                        dir, my, n, fi, rot1[0], rot2[0]);
                }
            }
        }
    }

    #[test]
    fn test_rotation_120_three_times_is_identity() {
        let table_120 = build_rotation_table(1);

        // Per-cell feature at center (should always be in bounds)
        let center = 10 * 21 + 10; // cell (0,0) = index 220
        let fi = (center * 11 + 3) as u16; // salmon at center
        let rot1 = rotate_features(&[fi], &table_120, 1).unwrap();
        let rot2 = rotate_features(&rot1, &table_120, 1).unwrap();
        let rot3 = rotate_features(&rot2, &table_120, 1).unwrap();
        assert_eq!(rot3[0], fi, "3x 120° rotation should be identity for cell features");

        // Pairwise feature
        for dir in 0..3 {
            for ps in 0..49 {
                let fi = (4961 + dir * 49 + ps) as u16;
                let r1 = rotate_features(&[fi], &table_120, 1).unwrap();
                let r2 = rotate_features(&r1, &table_120, 1).unwrap();
                let r3 = rotate_features(&r2, &table_120, 1).unwrap();
                assert_eq!(r3[0], fi,
                    "3x 120° not identity for pairwise dir={}, ps={}", dir, ps);
            }
        }
    }

    #[test]
    fn test_feature_block_boundaries() {
        // Verify the constants match between here and nnue.rs
        assert_eq!(crate::nnue::NUM_FEATURES, 7670);
        assert_eq!(crate::nnue::CELL_FEATURES, 4851);
        assert_eq!(crate::nnue::PHASE_FEATURES, 110);
        assert_eq!(crate::nnue::PAIR_FEATURES, 147);
        assert_eq!(crate::nnue::PATTERN_FEATURES, 89);
        assert_eq!(crate::nnue::BAG_FEATURES, 55);
        assert_eq!(crate::nnue::OPP_HAB_FEATURES, 55);
        assert_eq!(crate::nnue::ALLOWED_WL_FEATURES, 2205);
        assert_eq!(crate::nnue::WL_COUNT_EXT_FEATURES, 50);
        assert_eq!(crate::nnue::TERRAIN_PAIR_FEATURES, 108);
    }

    #[test]
    fn test_translation_preserves_pairwise_order() {
        // Translation (dir_shift=0) should NEVER swap pairwise pairs
        let table = build_translation_table(1, 0);

        for dir in 0..3 {
            let pair_state = 1 * 7 + 3; // bear-salmon
            let fi = (4961 + dir * 49 + pair_state) as u16;
            // translate_features uses rotate_features with dir_shift=0
            if let Some(trans) = rotate_features(&[fi], &table, 0) {
                let rel = trans[0] as usize - 4961;
                let new_ps = rel % 49;
                assert_eq!(new_ps, pair_state,
                    "Translation should not swap pairwise pair for dir {}", dir);
            }
        }
    }
}
