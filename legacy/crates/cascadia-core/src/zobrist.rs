use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use std::sync::LazyLock;

use crate::hex::GRID_SIZE;
use crate::types::{Terrain, Wildlife};

/// Zobrist hash keys for incremental board hashing.
/// Each (position, terrain) and (position, wildlife) pair has a unique random u64.
pub struct ZobristKeys {
    pub terrain: [[u64; 5]; GRID_SIZE],  // [position][terrain]
    pub wildlife: [[u64; 5]; GRID_SIZE], // [position][wildlife]
}

impl ZobristKeys {
    fn new() -> Self {
        let mut rng = StdRng::seed_from_u64(0xCACAD1A_CAFE);
        let mut keys = ZobristKeys {
            terrain: [[0u64; 5]; GRID_SIZE],
            wildlife: [[0u64; 5]; GRID_SIZE],
        };

        for pos in 0..GRID_SIZE {
            for t in 0..5 {
                keys.terrain[pos][t] = rng.gen();
            }
            for w in 0..5 {
                keys.wildlife[pos][w] = rng.gen();
            }
        }

        keys
    }

    #[inline(always)]
    pub fn terrain_key(&self, pos: usize, terrain: Terrain) -> u64 {
        self.terrain[pos][terrain as usize]
    }

    #[inline(always)]
    pub fn wildlife_key(&self, pos: usize, wildlife: Wildlife) -> u64 {
        self.wildlife[pos][wildlife as usize]
    }
}

pub static ZOBRIST: LazyLock<ZobristKeys> = LazyLock::new(ZobristKeys::new);
