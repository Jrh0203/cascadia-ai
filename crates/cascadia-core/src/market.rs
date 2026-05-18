use rand::seq::SliceRandom;
use rand::Rng;

use crate::types::{Terrain, TileData, Wildlife, WildlifeMask};

/// The tile bag containing all game tiles.
#[derive(Clone)]
pub struct TileBag {
    tiles: Vec<TileData>,
}

impl TileBag {
    /// Create and shuffle the standard 85-tile bag matching the official
    /// Cascadia tile set (from reference implementation).
    /// Terrain mapping: mountain, forest, desert=Prairie, swamp=Wetland, lake=River.
    pub fn new<R: Rng>(rng: &mut R) -> Self {
        use Terrain::*;
        use Wildlife::*;

        let mut tiles = Vec::with_capacity(85);
        let m = |a: &[Wildlife]| WildlifeMask::new(a);

        // === 25 Keystone tiles (tiles 16-40) ===
        // Mountain (16-20)
        tiles.push(TileData::single(Mountain, m(&[Hawk])));      // 16
        tiles.push(TileData::single(Mountain, m(&[Hawk])));      // 17
        tiles.push(TileData::single(Mountain, m(&[Bear])));      // 18
        tiles.push(TileData::single(Mountain, m(&[Elk])));       // 19
        tiles.push(TileData::single(Mountain, m(&[Elk])));       // 20
        // Forest (21-25)
        tiles.push(TileData::single(Forest, m(&[Bear])));        // 21
        tiles.push(TileData::single(Forest, m(&[Bear])));        // 22
        tiles.push(TileData::single(Forest, m(&[Elk])));         // 23
        tiles.push(TileData::single(Forest, m(&[Fox])));         // 24
        tiles.push(TileData::single(Forest, m(&[Fox])));         // 25
        // Prairie/Desert (26-30)
        tiles.push(TileData::single(Prairie, m(&[Elk])));        // 26
        tiles.push(TileData::single(Prairie, m(&[Elk])));        // 27
        tiles.push(TileData::single(Prairie, m(&[Fox])));        // 28
        tiles.push(TileData::single(Prairie, m(&[Salmon])));     // 29
        tiles.push(TileData::single(Prairie, m(&[Salmon])));     // 30
        // Wetland/Swamp (31-35)
        tiles.push(TileData::single(Wetland, m(&[Fox])));        // 31
        tiles.push(TileData::single(Wetland, m(&[Fox])));        // 32
        tiles.push(TileData::single(Wetland, m(&[Salmon])));     // 33
        tiles.push(TileData::single(Wetland, m(&[Salmon])));     // 34
        tiles.push(TileData::single(Wetland, m(&[Hawk])));       // 35
        // River/Lake (36-40)
        tiles.push(TileData::single(River, m(&[Hawk])));         // 36
        tiles.push(TileData::single(River, m(&[Hawk])));         // 37
        tiles.push(TileData::single(River, m(&[Salmon])));       // 38
        tiles.push(TileData::single(River, m(&[Bear])));         // 39
        tiles.push(TileData::single(River, m(&[Bear])));         // 40

        // === 60 Dual-terrain tiles (tiles 41-100) ===
        // Mountain/Forest (41-42, 56-59)
        tiles.push(TileData::dual(Mountain, Forest, m(&[Hawk, Bear, Elk])));  // 41
        tiles.push(TileData::dual(Mountain, Forest, m(&[Fox, Bear, Elk])));   // 42
        tiles.push(TileData::dual(Mountain, Forest, m(&[Hawk, Bear])));       // 56
        tiles.push(TileData::dual(Mountain, Forest, m(&[Hawk, Elk])));        // 57
        tiles.push(TileData::dual(Mountain, Forest, m(&[Bear, Fox])));        // 58
        tiles.push(TileData::dual(Mountain, Forest, m(&[Elk, Fox])));         // 59
        // Mountain/Prairie (43-44, 79-82)
        tiles.push(TileData::dual(Mountain, Prairie, m(&[Fox, Bear, Elk])));     // 43
        tiles.push(TileData::dual(Mountain, Prairie, m(&[Salmon, Fox, Bear]))); // 44
        tiles.push(TileData::dual(Mountain, Prairie, m(&[Hawk, Elk])));         // 79
        tiles.push(TileData::dual(Mountain, Prairie, m(&[Hawk, Fox])));         // 80
        tiles.push(TileData::dual(Mountain, Prairie, m(&[Bear, Salmon])));      // 81
        tiles.push(TileData::dual(Mountain, Prairie, m(&[Elk, Salmon])));       // 82
        // Mountain/Wetland (48-49, 83-86)
        tiles.push(TileData::dual(Mountain, Wetland, m(&[Fox, Hawk, Bear])));    // 48
        tiles.push(TileData::dual(Mountain, Wetland, m(&[Salmon, Bear, Elk]))); // 49
        tiles.push(TileData::dual(Mountain, Wetland, m(&[Hawk, Salmon])));       // 83
        tiles.push(TileData::dual(Mountain, Wetland, m(&[Bear, Salmon])));       // 84
        tiles.push(TileData::dual(Mountain, Wetland, m(&[Elk, Fox])));           // 85
        tiles.push(TileData::dual(Mountain, Wetland, m(&[Elk, Hawk])));          // 86
        // Forest/Prairie (45, 60-64)
        tiles.push(TileData::dual(Forest, Prairie, m(&[Salmon, Fox, Elk])));  // 45
        tiles.push(TileData::dual(Forest, Prairie, m(&[Bear, Elk])));         // 60
        tiles.push(TileData::dual(Forest, Prairie, m(&[Bear, Fox])));         // 61
        tiles.push(TileData::dual(Forest, Prairie, m(&[Elk, Fox])));          // 62
        tiles.push(TileData::dual(Forest, Prairie, m(&[Elk, Salmon])));       // 63
        tiles.push(TileData::dual(Forest, Prairie, m(&[Fox, Salmon])));       // 64
        // Forest/Wetland (50, 87-91)
        tiles.push(TileData::dual(Forest, Wetland, m(&[Salmon, Hawk, Elk]))); // 50
        tiles.push(TileData::dual(Forest, Wetland, m(&[Bear, Fox])));         // 87
        tiles.push(TileData::dual(Forest, Wetland, m(&[Bear, Salmon])));      // 88
        tiles.push(TileData::dual(Forest, Wetland, m(&[Elk, Salmon])));       // 89
        tiles.push(TileData::dual(Forest, Wetland, m(&[Elk, Hawk])));         // 90
        tiles.push(TileData::dual(Forest, Wetland, m(&[Fox, Hawk])));         // 91
        // Forest/River (52, 92-96)
        tiles.push(TileData::dual(Forest, River, m(&[Hawk, Fox, Elk])));      // 52
        tiles.push(TileData::dual(Forest, River, m(&[Bear, Salmon])));        // 92
        tiles.push(TileData::dual(Forest, River, m(&[Fox, Salmon])));         // 93
        tiles.push(TileData::dual(Forest, River, m(&[Elk, Hawk])));           // 94
        tiles.push(TileData::dual(Forest, River, m(&[Elk, Bear])));           // 95
        tiles.push(TileData::dual(Forest, River, m(&[Fox, Bear])));           // 96
        // Prairie/Wetland (46-47, 65-68)
        tiles.push(TileData::dual(Prairie, Wetland, m(&[Salmon, Fox, Elk])));  // 46
        tiles.push(TileData::dual(Prairie, Wetland, m(&[Salmon, Fox, Hawk]))); // 47
        tiles.push(TileData::dual(Prairie, Wetland, m(&[Elk, Fox])));          // 65
        tiles.push(TileData::dual(Prairie, Wetland, m(&[Elk, Salmon])));       // 66
        tiles.push(TileData::dual(Prairie, Wetland, m(&[Fox, Hawk])));         // 67
        tiles.push(TileData::dual(Prairie, Wetland, m(&[Salmon, Hawk])));      // 68
        // Prairie/River (54-55, 97-100)
        tiles.push(TileData::dual(Prairie, River, m(&[Salmon, Fox, Bear])));   // 54
        tiles.push(TileData::dual(Prairie, River, m(&[Fox, Hawk, Bear])));     // 55
        tiles.push(TileData::dual(Prairie, River, m(&[Elk, Salmon])));         // 97
        tiles.push(TileData::dual(Prairie, River, m(&[Elk, Hawk])));           // 98
        tiles.push(TileData::dual(Prairie, River, m(&[Fox, Hawk])));           // 99
        tiles.push(TileData::dual(Prairie, River, m(&[Fox, Bear])));           // 100
        // Wetland/River (51, 69-73)
        tiles.push(TileData::dual(Wetland, River, m(&[Salmon, Hawk, Bear]))); // 51
        tiles.push(TileData::dual(Wetland, River, m(&[Fox, Salmon])));        // 69
        tiles.push(TileData::dual(Wetland, River, m(&[Fox, Hawk])));          // 70
        tiles.push(TileData::dual(Wetland, River, m(&[Salmon, Hawk])));       // 71
        tiles.push(TileData::dual(Wetland, River, m(&[Salmon, Bear])));       // 72
        tiles.push(TileData::dual(Wetland, River, m(&[Hawk, Bear])));         // 73
        // River/Mountain (53, 74-78)
        tiles.push(TileData::dual(River, Mountain, m(&[Salmon, Hawk, Bear]))); // 53
        tiles.push(TileData::dual(River, Mountain, m(&[Salmon, Hawk])));       // 74
        tiles.push(TileData::dual(River, Mountain, m(&[Salmon, Bear])));       // 75
        tiles.push(TileData::dual(River, Mountain, m(&[Hawk, Bear])));         // 76
        tiles.push(TileData::dual(River, Mountain, m(&[Hawk, Elk])));          // 77
        tiles.push(TileData::dual(River, Mountain, m(&[Bear, Elk])));          // 78

        tiles.shuffle(rng);
        TileBag { tiles }
    }

    pub fn draw(&mut self) -> Option<TileData> {
        self.tiles.pop()
    }

    /// Return a tile to the bottom of the bag (used for independent draft leftovers).
    pub fn return_tile(&mut self, tile: TileData) {
        self.tiles.insert(0, tile);
    }

    pub fn remaining(&self) -> usize {
        self.tiles.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tiles.is_empty()
    }

    /// Shuffle remaining tiles in the bag (used to prevent future-peeking in search).
    pub fn shuffle<R: Rng>(&mut self, rng: &mut R) {
        self.tiles.shuffle(rng);
    }

    /// Returns (terrain_distribution, wildlife_capacity) summaries used by NNUE features.
    /// - terrain_distribution[t] = count of remaining tiles whose primary OR secondary terrain == t
    /// - wildlife_capacity[w] = count of remaining tiles whose allowed mask includes wildlife w
    pub fn feature_distributions(&self) -> ([u8; 5], [u8; 5]) {
        use crate::types::Wildlife;
        let mut terrain = [0u8; 5];
        let mut wildlife = [0u8; 5];
        for tile in &self.tiles {
            terrain[tile.terrain1 as usize] = terrain[tile.terrain1 as usize].saturating_add(1);
            if let Some(t2) = tile.terrain2 {
                terrain[t2 as usize] = terrain[t2 as usize].saturating_add(1);
            }
            for w in Wildlife::ALL {
                if tile.allowed.contains(w) {
                    wildlife[w as usize] = wildlife[w as usize].saturating_add(1);
                }
            }
        }
        (terrain, wildlife)
    }

    /// Joint terrain × wildlife distribution: joint[t][w] = count of remaining
    /// tiles whose primary OR secondary terrain == t AND whose allowed mask
    /// includes wildlife w. Each tile contributes to up to 2 (t,w) cells per
    /// allowed wildlife (once for terrain1, once for terrain2 if dual). Used by
    /// the v5-feat NNUE block to give the value function a joint signal for
    /// "how many drafts that benefit me are still available?".
    pub fn joint_distribution(&self) -> [[u8; 5]; 5] {
        use crate::types::Wildlife;
        let mut joint = [[0u8; 5]; 5];
        for tile in &self.tiles {
            let t1 = tile.terrain1 as usize;
            let t2 = tile.terrain2.map(|t| t as usize);
            for w in Wildlife::ALL {
                if !tile.allowed.contains(w) { continue; }
                let wi = w as usize;
                joint[t1][wi] = joint[t1][wi].saturating_add(1);
                if let Some(ti) = t2 {
                    joint[ti][wi] = joint[ti][wi].saturating_add(1);
                }
            }
        }
        joint
    }
}

/// The wildlife token bag.
#[derive(Clone)]
pub struct WildlifeBag {
    tokens: Vec<Wildlife>,
    /// Tokens set aside during mulligan, returned at end of turn.
    deferred_returns: Vec<Wildlife>,
}

impl WildlifeBag {
    /// Create the standard wildlife bag (20 of each type = 100 tokens).
    pub fn new<R: Rng>(rng: &mut R) -> Self {
        let mut tokens = Vec::with_capacity(100);
        for &w in &Wildlife::ALL {
            for _ in 0..20 {
                tokens.push(w);
            }
        }
        tokens.shuffle(rng);
        WildlifeBag {
            tokens,
            deferred_returns: Vec::new(),
        }
    }

    pub fn draw(&mut self) -> Option<Wildlife> {
        self.tokens.pop()
    }

    pub fn remaining(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Shuffle remaining tokens in the bag (used to prevent future-peeking in search).
    pub fn shuffle<R: Rng>(&mut self, rng: &mut R) {
        self.tokens.shuffle(rng);
    }

    /// Return a wildlife token to the bag immediately.
    pub fn return_token(&mut self, w: Wildlife) {
        self.tokens.push(w);
    }

    /// Set aside a token to be returned at end of turn.
    pub fn defer_return(&mut self, w: Wildlife) {
        self.deferred_returns.push(w);
    }

    /// Flush deferred returns back into the bag (call at end of turn).
    pub fn flush_deferred(&mut self) {
        self.tokens.append(&mut self.deferred_returns);
    }
}

/// A draft pair in the market: a tile paired with a wildlife token.
#[derive(Debug, Clone, Copy)]
pub struct MarketPair {
    pub tile: TileData,
    pub wildlife: Wildlife,
}

/// The market of 4 tile+wildlife pairs available for drafting.
#[derive(Clone)]
pub struct Market {
    pub pairs: [Option<MarketPair>; 4],
}

impl Market {
    /// Initialize the market by drawing 4 pairs from the bags.
    pub fn new(tile_bag: &mut TileBag, wildlife_bag: &mut WildlifeBag) -> Self {
        let mut pairs = [None; 4];
        for slot in pairs.iter_mut() {
            if let (Some(tile), Some(wildlife)) = (tile_bag.draw(), wildlife_bag.draw()) {
                *slot = Some(MarketPair { tile, wildlife });
            }
        }
        Market { pairs }
    }

    /// Draft a pair from the market by index (0-3). Returns the pair and leaves the slot empty.
    pub fn draft(&mut self, index: usize) -> Option<MarketPair> {
        self.pairs[index].take()
    }

    /// Refill empty slots from the bags. Does NOT auto-replace 3-of-a-kind
    /// — that's now an optional player choice (use `has_3_of_kind` + `replace_3_of_kind`).
    /// Caller must ensure deferred wildlife is flushed BEFORE calling refill at
    /// end of turn (so mulligan tokens are available).
    pub fn refill(&mut self, tile_bag: &mut TileBag, wildlife_bag: &mut WildlifeBag) {
        for slot in self.pairs.iter_mut() {
            if slot.is_none() {
                if let (Some(tile), Some(wildlife)) = (tile_bag.draw(), wildlife_bag.draw()) {
                    *slot = Some(MarketPair { tile, wildlife });
                }
            }
        }
    }

    /// Returns the wildlife type if 3+ slots have the same wildlife.
    pub fn has_3_of_kind(&self) -> Option<Wildlife> {
        let mut counts = [0u8; 5];
        for pair in self.pairs.iter().flatten() {
            counts[pair.wildlife as usize] += 1;
        }
        counts.iter().position(|&c| c >= 3)
            .and_then(|ot| Wildlife::from_u8(ot as u8))
    }

    /// Replace 3-of-a-kind wildlife tokens with new draws from the bag.
    /// Old tokens are deferred (returned at end of turn). Free action, once per turn.
    pub fn replace_3_of_kind(&mut self, wildlife_bag: &mut WildlifeBag) -> bool {
        let overflow = match self.has_3_of_kind() {
            Some(w) => w,
            None => return false,
        };
        for slot in self.pairs.iter_mut() {
            if let Some(pair) = slot {
                if pair.wildlife == overflow {
                    wildlife_bag.defer_return(pair.wildlife);
                    if let Some(new_wildlife) = wildlife_bag.draw() {
                        pair.wildlife = new_wildlife;
                    }
                }
            }
        }
        true
    }

    #[allow(dead_code)]
    fn check_wildlife_overflow(&mut self, wildlife_bag: &mut WildlifeBag) {
        let mut counts = [0u8; 5];
        for pair in self.pairs.iter().flatten() {
            counts[pair.wildlife as usize] += 1;
        }

        // Find which wildlife type has 3+
        let overflow_type = counts.iter().position(|&c| c >= 3);
        if let Some(ot) = overflow_type {
            let overflow_wildlife = Wildlife::from_u8(ot as u8).unwrap();
            for slot in self.pairs.iter_mut() {
                if let Some(pair) = slot {
                    if pair.wildlife == overflow_wildlife {
                        wildlife_bag.defer_return(pair.wildlife);
                        if let Some(new_wildlife) = wildlife_bag.draw() {
                            pair.wildlife = new_wildlife;
                        }
                    }
                }
            }
        }
    }

    /// Mulligan: spend a nature token to replace ALL 4 wildlife tokens.
    /// Old tokens are deferred (returned at end of turn).
    /// Returns true if successful.
    pub fn mulligan_all_wildlife(&mut self, wildlife_bag: &mut WildlifeBag) -> bool {
        // Set aside all current wildlife tokens
        for slot in self.pairs.iter_mut() {
            if let Some(pair) = slot {
                wildlife_bag.defer_return(pair.wildlife);
                if let Some(new_wildlife) = wildlife_bag.draw() {
                    pair.wildlife = new_wildlife;
                }
            }
        }
        true
    }

    /// Draft a tile from one slot and wildlife from another (costs a nature token).
    /// The "leftover" items (wildlife from tile_index's pair, tile from
    /// wildlife_index's pair) are returned to their respective bags so they
    /// stay in play. Both slots are emptied for refill.
    pub fn draft_independent(
        &mut self,
        tile_index: usize,
        wildlife_index: usize,
        tile_bag: &mut TileBag,
        wildlife_bag: &mut WildlifeBag,
    ) -> Option<(TileData, Wildlife)> {
        if tile_index == wildlife_index {
            let pair = self.pairs[tile_index].take()?;
            return Some((pair.tile, pair.wildlife));
        }

        let tile_pair = self.pairs[tile_index].take()?;
        let wildlife_pair = self.pairs[wildlife_index].take()?;

        // Return leftover items to their bags
        wildlife_bag.return_token(tile_pair.wildlife);
        tile_bag.return_tile(wildlife_pair.tile);

        Some((tile_pair.tile, wildlife_pair.wildlife))
    }

    /// Check if 3+ of the same wildlife are showing (for UI indication).
    pub fn has_overflow(&self) -> bool {
        let mut counts = [0u8; 5];
        for pair in self.pairs.iter().flatten() {
            counts[pair.wildlife as usize] += 1;
        }
        counts.iter().any(|&c| c >= 3)
    }

    /// Get available (non-empty) market slots.
    pub fn available(&self) -> impl Iterator<Item = (usize, &MarketPair)> {
        self.pairs
            .iter()
            .enumerate()
            .filter_map(|(i, slot)| slot.as_ref().map(|p| (i, p)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn tile_bag_has_85_tiles() {
        let mut rng = StdRng::seed_from_u64(42);
        let bag = TileBag::new(&mut rng);
        assert_eq!(bag.remaining(), 85);
    }

    #[test]
    fn tile_distribution_matches_official() {
        let mut rng = StdRng::seed_from_u64(42);
        let bag = TileBag::new(&mut rng);

        let mut keystones = 0;
        let mut animal_per_habitat = [[0u32; 5]; 5]; // [habitat][animal]
        let mut animal_total = [0u32; 5];
        let mut habitat_total = [0u32; 5];

        for tile in &bag.tiles {
            if tile.keystone {
                keystones += 1;
            }

            let habitats: Vec<usize> = {
                let mut h = vec![tile.terrain1 as usize];
                if let Some(t2) = tile.terrain2 {
                    h.push(t2 as usize);
                }
                h
            };

            for &h in &habitats {
                habitat_total[h] += 1;
                for a in 0..5 {
                    if tile.allowed.contains(Wildlife::from_u8(a as u8).unwrap()) {
                        animal_per_habitat[h][a] += 1;
                    }
                }
            }

            for a in 0..5 {
                if tile.allowed.contains(Wildlife::from_u8(a as u8).unwrap()) {
                    animal_total[a] += 1;
                }
            }
        }

        assert_eq!(keystones, 25);
        // Each animal appears on 32 tiles
        for a in 0..5 {
            assert_eq!(animal_total[a], 32, "animal {} total", a);
        }
        // Each habitat appears on 29 tiles (5 keystone + 24 dual)
        for h in 0..5 {
            assert_eq!(habitat_total[h], 29, "habitat {} total", h);
        }

        // Animal per habitat, indexed by Terrain enum order:
        // Forest=0, Prairie=1, Wetland=2, Mountain=3, River=4
        let expected: [[u32; 5]; 5] = [
            // Bear  Elk  Salmon Hawk  Fox
            [13, 15, 8, 8, 14],   // Forest
            [8, 14, 14, 8, 16],   // Prairie
            [8, 9, 16, 14, 12],   // Wetland
            [15, 14, 9, 14, 8],   // Mountain
            [15, 7, 12, 15, 9],   // River
        ];

        for h in 0..5 {
            for a in 0..5 {
                assert_eq!(
                    animal_per_habitat[h][a], expected[h][a],
                    "habitat {} animal {} mismatch", h, a
                );
            }
        }
    }

    #[test]
    fn wildlife_bag_has_100_tokens() {
        let mut rng = StdRng::seed_from_u64(42);
        let bag = WildlifeBag::new(&mut rng);
        assert_eq!(bag.remaining(), 100);
    }

    #[test]
    fn market_has_4_pairs() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut tile_bag = TileBag::new(&mut rng);
        let mut wildlife_bag = WildlifeBag::new(&mut rng);
        let market = Market::new(&mut tile_bag, &mut wildlife_bag);
        assert_eq!(market.available().count(), 4);
    }

    #[test]
    fn draft_and_refill() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut tile_bag = TileBag::new(&mut rng);
        let mut wildlife_bag = WildlifeBag::new(&mut rng);
        let mut market = Market::new(&mut tile_bag, &mut wildlife_bag);

        let pair = market.draft(0);
        assert!(pair.is_some());
        assert_eq!(market.available().count(), 3);

        market.refill(&mut tile_bag, &mut wildlife_bag);
        assert_eq!(market.available().count(), 4);
    }
}
