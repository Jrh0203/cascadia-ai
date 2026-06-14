pub mod bear;
pub mod elk;
pub mod fox;
pub mod hawk;
pub(crate) mod matching;
pub mod salmon;

use crate::board::Board;
use crate::types::{ScoringCardVariant, ScoringCards, Wildlife};

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
    use ScoringCardVariant::*;
    match (wildlife, variant) {
        (Wildlife::Bear, A) => bear::score_a(board),
        (Wildlife::Bear, B) => bear::score_b(board),
        (Wildlife::Bear, C) => bear::score_c(board),
        (Wildlife::Bear, D) => bear::score_d(board),

        (Wildlife::Elk, A) => elk::score_a(board),
        (Wildlife::Elk, B) => elk::score_b(board),
        (Wildlife::Elk, C) => elk::score_c(board),
        (Wildlife::Elk, D) => elk::score_d(board),

        (Wildlife::Salmon, A) => salmon::score_a(board),
        (Wildlife::Salmon, B) => salmon::score_b(board),
        (Wildlife::Salmon, C) => salmon::score_c(board),
        (Wildlife::Salmon, D) => salmon::score_d(board),

        (Wildlife::Hawk, A) => hawk::score_a(board),
        (Wildlife::Hawk, B) => hawk::score_b(board),
        (Wildlife::Hawk, C) => hawk::score_c(board),
        (Wildlife::Hawk, D) => hawk::score_d(board),

        (Wildlife::Fox, A) => fox::score_a(board),
        (Wildlife::Fox, B) => fox::score_b(board),
        (Wildlife::Fox, C) => fox::score_c(board),
        (Wildlife::Fox, D) => fox::score_d(board),
    }
}
