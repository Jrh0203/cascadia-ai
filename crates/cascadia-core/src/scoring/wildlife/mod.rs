pub mod bear;
pub mod elk;
pub mod salmon;
pub mod hawk;
pub mod fox;

use crate::board::Board;
use crate::types::{ScoringCards, ScoringCardVariant, Wildlife};

/// Score all 5 wildlife types. Returns [bear, elk, salmon, hawk, fox] scores.
pub fn score_all_wildlife(board: &Board, cards: &ScoringCards) -> [u16; 5] {
    [
        score_wildlife(board, Wildlife::Bear, cards.variant_for(Wildlife::Bear)),
        score_wildlife(board, Wildlife::Elk, cards.variant_for(Wildlife::Elk)),
        score_wildlife(board, Wildlife::Salmon, cards.variant_for(Wildlife::Salmon)),
        score_wildlife(board, Wildlife::Hawk, cards.variant_for(Wildlife::Hawk)),
        score_wildlife(board, Wildlife::Fox, cards.variant_for(Wildlife::Fox)),
    ]
}

/// Score a single wildlife type.
pub fn score_wildlife(board: &Board, wildlife: Wildlife, variant: ScoringCardVariant) -> u16 {
    match (wildlife, variant) {
        (Wildlife::Bear, ScoringCardVariant::A) => bear::score_a(board),
        (Wildlife::Elk, ScoringCardVariant::A) => elk::score_a(board),
        (Wildlife::Salmon, ScoringCardVariant::A) => salmon::score_a(board),
        (Wildlife::Hawk, ScoringCardVariant::A) => hawk::score_a(board),
        (Wildlife::Fox, ScoringCardVariant::A) => fox::score_a(board),
        // B/C/D variants — to be implemented
        _ => 0,
    }
}
