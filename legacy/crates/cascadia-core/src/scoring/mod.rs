pub mod habitat;
pub mod wildlife;

use crate::board::Board;
use crate::types::{ScoringCards, Terrain};

/// Complete score breakdown for a player.
#[derive(Debug, Clone, Copy, Default)]
pub struct ScoreBreakdown {
    pub habitat: [u16; 5],
    pub wildlife: [u16; 5],
    pub nature_tokens: u16,
    pub habitat_bonus: [u8; 5],
    pub total: u16,
}

impl ScoreBreakdown {
    /// Compute the full score for a board (single-player, no habitat bonuses).
    pub fn compute(board: &mut Board, cards: &ScoringCards) -> Self {
        let mut breakdown = ScoreBreakdown::default();

        // Habitat scoring: largest group per terrain
        for terrain in Terrain::ALL {
            breakdown.habitat[terrain as usize] = board.largest_group[terrain as usize];
        }

        // Wildlife scoring
        breakdown.wildlife = wildlife::score_all_wildlife(board, cards);

        // Nature tokens
        breakdown.nature_tokens = board.nature_tokens as u16;

        // Total
        breakdown.total = breakdown.habitat.iter().sum::<u16>()
            + breakdown.wildlife.iter().sum::<u16>()
            + breakdown.nature_tokens;

        breakdown
    }

    /// Compute with habitat majority bonuses (multiplayer).
    pub fn compute_with_bonuses(boards: &mut [Board], cards: &ScoringCards, player: usize) -> Self {
        let mut breakdown = Self::compute(&mut boards[player], cards);

        // Compute habitat bonuses
        let num_players = boards.len();
        for terrain in Terrain::ALL {
            let ti = terrain as usize;
            let my_size = boards[player].largest_group[ti];
            let mut rank = 0usize; // how many players have strictly larger
            let mut tied = 0usize; // how many players have the same size

            for (i, board) in boards.iter().enumerate() {
                if i == player {
                    continue;
                }
                let their_size = board.largest_group[ti];
                if their_size > my_size {
                    rank += 1;
                } else if their_size == my_size {
                    tied += 1;
                }
            }

            let bonus = if num_players == 2 {
                if rank == 0 && tied == 0 {
                    2 // sole largest
                } else if rank == 0 {
                    1 // tied for largest
                } else {
                    0
                }
            } else {
                // 3-4 players
                if rank == 0 && tied == 0 {
                    3 // sole largest
                } else if rank == 0 {
                    2 // tied for largest: 2 pts each
                } else if rank == 1 && tied == 0 {
                    1 // sole second largest
                } else {
                    0 // tied for second or worse
                }
            };

            breakdown.habitat_bonus[ti] = bonus;
        }

        breakdown.total += breakdown
            .habitat_bonus
            .iter()
            .map(|&b| b as u16)
            .sum::<u16>();
        breakdown
    }

    /// Sum of wildlife scores across all 5 species (bear/elk/salmon/hawk/fox).
    /// This is the "wildlife_total" used as a split-value-head training target.
    #[inline]
    pub fn wildlife_total(&self) -> u16 {
        self.wildlife.iter().sum()
    }

    /// Everything except wildlife: sum(habitat) + nature_tokens.
    /// Note: does NOT include habitat_bonus (that's only populated by compute_with_bonuses
    /// in 4-player context; single-player training uses Self::compute which leaves it at 0).
    /// This is the "habitat+tokens" training target for the non-wildlife value head.
    #[inline]
    pub fn non_wildlife_total(&self) -> u16 {
        let hab: u16 = self.habitat.iter().sum();
        let bonus: u16 = self.habitat_bonus.iter().map(|&b| b as u16).sum();
        hab + self.nature_tokens + bonus
    }
}
