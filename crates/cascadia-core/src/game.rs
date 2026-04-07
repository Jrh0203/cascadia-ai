use rand::seq::SliceRandom;
use rand::Rng;

use crate::board::Board;
use crate::hex::HexCoord;
use crate::market::{Market, TileBag, WildlifeBag};
use crate::scoring::ScoreBreakdown;
use crate::types::{ScoringCards, Terrain, TileData, Wildlife, WildlifeMask};

/// A starter tile with its data and rotation.
struct StarterHex {
    tile: TileData,
    rotation: u8,
}

/// The 5 official starting tile clusters from the reference implementation.
/// Layout: keystone at (0,0), dual tiles at (0,1) and (1,0) — clockwise from keystone.
/// Dual tiles are rotated so the terrain split faces the center of the cluster.
///
/// For tile at (1,0): inward edges face directions 3 (W) and 4 (SW).
///   Rotation 4 puts the split between directions 3 and 4.
/// For tile at (0,1): inward edges face directions 1 (NE) and 2 (NW).
///   Rotation 5 puts the split between directions 1 and 2.
fn all_starter_clusters() -> [[(HexCoord, StarterHex); 3]; 5] {
    use Terrain::*;
    use Wildlife::*;
    let m = |a: &[Wildlife]| WildlifeMask::new(a);

    // Positions: keystone at (0,0), first dual at (0,1), second dual at (1,0)
    let p0 = HexCoord::new(0, 0);
    let p1 = HexCoord::new(0, 1); // clockwise from keystone
    let p2 = HexCoord::new(1, 0); // clockwise from p1

    // Rotations for inward-facing split
    let rot_p1: u8 = 5; // split faces directions 1-2 (toward center)
    let rot_p2: u8 = 4; // split faces directions 3-4 (toward center)

    [
        // Cluster 1: Mountain(bear) + Forest/Wetland(hawk,fox,elk) + Prairie/River(salmon,bear)
        [
            (p0, StarterHex { tile: TileData::single(Mountain, m(&[Bear])), rotation: 0 }),
            (p1, StarterHex { tile: TileData::dual(Forest, Wetland, m(&[Hawk, Fox, Elk])), rotation: rot_p1 }),
            (p2, StarterHex { tile: TileData::dual(Prairie, River, m(&[Salmon, Bear])), rotation: rot_p2 }),
        ],
        // Cluster 2: Wetland(hawk) + Forest/River(salmon,hawk,elk) + Mountain/Prairie(bear,fox)
        [
            (p0, StarterHex { tile: TileData::single(Wetland, m(&[Hawk])), rotation: 0 }),
            (p1, StarterHex { tile: TileData::dual(Forest, River, m(&[Salmon, Hawk, Elk])), rotation: rot_p1 }),
            (p2, StarterHex { tile: TileData::dual(Mountain, Prairie, m(&[Bear, Fox])), rotation: rot_p2 }),
        ],
        // Cluster 3: Prairie(fox) + Wetland/River(salmon,fox,hawk) + Mountain/Forest(bear,elk)
        [
            (p0, StarterHex { tile: TileData::single(Prairie, m(&[Fox])), rotation: 0 }),
            (p1, StarterHex { tile: TileData::dual(Wetland, River, m(&[Salmon, Fox, Hawk])), rotation: rot_p1 }),
            (p2, StarterHex { tile: TileData::dual(Mountain, Forest, m(&[Bear, Elk])), rotation: rot_p2 }),
        ],
        // Cluster 4: Forest(elk) + River/Mountain(hawk,bear,elk) + Prairie/Wetland(fox,salmon)
        [
            (p0, StarterHex { tile: TileData::single(Forest, m(&[Elk])), rotation: 0 }),
            (p1, StarterHex { tile: TileData::dual(River, Mountain, m(&[Hawk, Bear, Elk])), rotation: rot_p1 }),
            (p2, StarterHex { tile: TileData::dual(Prairie, Wetland, m(&[Fox, Salmon])), rotation: rot_p2 }),
        ],
        // Cluster 5: River(salmon) + Forest/Prairie(salmon,bear,elk) + Mountain/Wetland(fox,hawk)
        [
            (p0, StarterHex { tile: TileData::single(River, m(&[Salmon])), rotation: 0 }),
            (p1, StarterHex { tile: TileData::dual(Forest, Prairie, m(&[Salmon, Bear, Elk])), rotation: rot_p1 }),
            (p2, StarterHex { tile: TileData::dual(Mountain, Wetland, m(&[Fox, Hawk])), rotation: rot_p2 }),
        ],
    ]
}

/// A player's move: which market slot to draft, where to place the tile,
/// and optionally where to place the wildlife token.
#[derive(Debug, Clone, Copy)]
pub struct PlayerMove {
    pub market_index: usize,
    pub tile_coord: HexCoord,
    pub rotation: u8, // 0-5, rotation of the placed tile
    pub wildlife_placement: Option<usize>, // grid index for wildlife, None to skip
}

/// The full game state.
#[derive(Clone)]
pub struct GameState {
    pub boards: Vec<Board>,
    pub market: Market,
    pub tile_bag: TileBag,
    pub wildlife_bag: WildlifeBag,
    pub scoring_cards: ScoringCards,
    pub current_player: usize,
    pub turns_remaining: u8,
    pub num_players: usize,
    /// Whether the free 3-of-a-kind replacement has been used this turn.
    pub overflow_used_this_turn: bool,
}

impl GameState {
    /// Create a new game with the given number of players.
    pub fn new<R: Rng>(num_players: usize, scoring_cards: ScoringCards, rng: &mut R) -> Self {
        assert!((1..=4).contains(&num_players));

        let mut tile_bag = TileBag::new(rng);
        let mut wildlife_bag = WildlifeBag::new(rng);

        // Shuffle and assign starter clusters to players
        let mut clusters = all_starter_clusters();
        clusters.shuffle(rng);

        let mut boards = Vec::with_capacity(num_players);
        for i in 0..num_players {
            let mut board = Board::new();
            for (coord, hex) in &clusters[i] {
                board.place_tile(*coord, hex.tile, hex.rotation);
            }
            boards.push(board);
        }

        let market = Market::new(&mut tile_bag, &mut wildlife_bag);

        // Standard game: 20 turns per player
        let turns_remaining = 20 * num_players as u8;

        GameState {
            boards,
            market,
            tile_bag,
            wildlife_bag,
            scoring_cards,
            current_player: 0,
            turns_remaining,
            num_players,
            overflow_used_this_turn: false,
        }
    }

    /// Execute a player's move.
    pub fn execute_move(&mut self, player_move: PlayerMove) -> bool {
        if self.turns_remaining == 0 {
            return false;
        }

        let board = &mut self.boards[self.current_player];

        // Draft from market
        let pair = match self.market.draft(player_move.market_index) {
            Some(p) => p,
            None => return false,
        };

        // Place tile
        if board.place_tile(player_move.tile_coord, pair.tile, player_move.rotation).is_none() {
            return false;
        }

        // Place wildlife (optional)
        if let Some(wildlife_idx) = player_move.wildlife_placement {
            board.place_wildlife(wildlife_idx, pair.wildlife);
        }

        // End of turn: flush deferred wildlife back to the bag FIRST,
        // then refill the market. This ensures mulligan tokens are available
        // for the refill (the game rule: deferred tokens return at end of turn
        // BEFORE the next refill happens).
        self.wildlife_bag.flush_deferred();
        self.market
            .refill(&mut self.tile_bag, &mut self.wildlife_bag);

        // Advance to next player
        self.current_player = (self.current_player + 1) % self.num_players;
        self.turns_remaining -= 1;
        self.overflow_used_this_turn = false;

        true
    }

    /// Execute a move spending a nature token to pick any tile and any wildlife
    /// independently from the market (they don't need to be from the same pair).
    pub fn execute_independent_move(
        &mut self,
        tile_market_index: usize,
        wildlife_market_index: usize,
        tile_coord: HexCoord,
        rotation: u8,
        wildlife_placement: Option<usize>,
    ) -> bool {
        if self.turns_remaining == 0 {
            return false;
        }

        let board = &mut self.boards[self.current_player];
        if board.nature_tokens == 0 {
            return false;
        }
        board.nature_tokens -= 1;

        let (tile, wildlife) = match self.market.draft_independent(
            tile_market_index, wildlife_market_index,
            &mut self.tile_bag, &mut self.wildlife_bag,
        ) {
            Some(pair) => pair,
            None => return false,
        };

        if board.place_tile(tile_coord, tile, rotation).is_none() {
            return false;
        }

        if let Some(wildlife_idx) = wildlife_placement {
            board.place_wildlife(wildlife_idx, wildlife);
        }

        // Flush deferred wildlife first, then refill (see execute_move comment)
        self.wildlife_bag.flush_deferred();
        self.market.refill(&mut self.tile_bag, &mut self.wildlife_bag);
        self.current_player = (self.current_player + 1) % self.num_players;
        self.turns_remaining -= 1;
        self.overflow_used_this_turn = false;

        true
    }

    /// Spend a nature token to mulligan all 4 wildlife tokens in the market.
    /// Returns true if successful (player has tokens and mulligan worked).
    /// Can be called multiple times per turn — deferred tokens only return at end of turn.
    pub fn mulligan_wildlife(&mut self) -> bool {
        let board = &mut self.boards[self.current_player];
        if board.nature_tokens == 0 {
            return false;
        }
        board.nature_tokens -= 1;
        self.market.mulligan_all_wildlife(&mut self.wildlife_bag);
        true
    }

    /// Check if the market has 3+ of the same wildlife AND the free replace hasn't been used this turn.
    pub fn can_replace_overflow(&self) -> Option<crate::types::Wildlife> {
        if self.overflow_used_this_turn { return None; }
        self.market.has_3_of_kind()
    }

    /// Take the free 3-of-a-kind replacement (once per turn, before the move).
    pub fn replace_overflow(&mut self) -> bool {
        if self.overflow_used_this_turn { return false; }
        if self.market.replace_3_of_kind(&mut self.wildlife_bag) {
            self.overflow_used_this_turn = true;
            true
        } else {
            false
        }
    }

    pub fn is_game_over(&self) -> bool {
        // Game ends when turns run out OR when the market is empty (can't make any moves).
        // The latter can happen if the bag runs out (e.g., after aggressive mulliganing).
        self.turns_remaining == 0 || self.market.available().next().is_none()
    }

    /// Shuffle remaining tiles and wildlife in bags to prevent future-peeking in search.
    pub fn shuffle_bags<R: Rng>(&mut self, rng: &mut R) {
        self.tile_bag.shuffle(rng);
        self.wildlife_bag.shuffle(rng);
    }

    /// Compute final scores for all players.
    pub fn final_scores(&mut self) -> Vec<ScoreBreakdown> {
        if self.num_players == 1 {
            vec![ScoreBreakdown::compute(
                &mut self.boards[0],
                &self.scoring_cards,
            )]
        } else {
            (0..self.num_players)
                .map(|p| {
                    ScoreBreakdown::compute_with_bonuses(
                        &mut self.boards,
                        &self.scoring_cards,
                        p,
                    )
                })
                .collect()
        }
    }

    pub fn current_board(&self) -> &Board {
        &self.boards[self.current_player]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn new_game_setup() {
        let mut rng = StdRng::seed_from_u64(42);
        let game = GameState::new(2, ScoringCards::all_a(), &mut rng);
        assert_eq!(game.num_players, 2);
        assert_eq!(game.turns_remaining, 40);
        assert_eq!(game.current_player, 0);
        assert_eq!(game.boards.len(), 2);
        // Each board starts with 3 tiles
        assert_eq!(game.boards[0].tile_count, 3);
        assert_eq!(game.boards[1].tile_count, 3);
    }

    #[test]
    fn game_advances_player() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut game = GameState::new(2, ScoringCards::all_a(), &mut rng);

        let frontier = game.current_board().frontier();
        let coord = HexCoord::from_index(frontier[0] as usize);

        let mv = PlayerMove {
            market_index: 0,
            tile_coord: coord,
            rotation: 0,
            wildlife_placement: None,
        };

        assert!(game.execute_move(mv));
        assert_eq!(game.current_player, 1);
        assert_eq!(game.turns_remaining, 39);
    }
}
