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

/// Score one wildlife category after hypothetically placing a token at `index`.
///
/// Wildlife cards read only wildlife occupancy and the per-type position
/// lists. They do not depend on tile terrain, habitat union-find state, or
/// Nature Tokens. Updating only those two wildlife representations therefore
/// produces the same score as a full tile/wildlife placement while avoiding
/// unrelated board mutations in search hot paths.
pub fn score_wildlife_after_placement(
    board: &mut Board,
    wildlife: Wildlife,
    variant: ScoringCardVariant,
    index: usize,
) -> u16 {
    let previous_cell = board.grid.get(index);
    assert!(
        previous_cell.placed_wildlife().is_none(),
        "hypothetical wildlife placement requires an unoccupied cell"
    );
    board.grid.set(index, previous_cell.with_wildlife(wildlife));
    board.wildlife_positions[wildlife as usize].push(index as u16);
    let score = score_wildlife(board, wildlife, variant);
    let removed = board.wildlife_positions[wildlife as usize]
        .pop()
        .expect("hypothetical wildlife position was appended");
    debug_assert_eq!(removed, index as u16);
    board.grid.set(index, previous_cell);
    score
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        hex::HexCoord,
        types::{ScoringCardVariant, Terrain, TileData, WildlifeMask},
    };

    fn any_tile() -> TileData {
        TileData::dual(
            Terrain::Forest,
            Terrain::Prairie,
            WildlifeMask::new(&Wildlife::ALL),
        )
    }

    fn representative_board() -> Board {
        let mut board = Board::new();
        let placements = [
            (0, 0, Some(Wildlife::Bear)),
            (1, 0, Some(Wildlife::Bear)),
            (0, 1, Some(Wildlife::Elk)),
            (1, -1, Some(Wildlife::Elk)),
            (2, -1, Some(Wildlife::Salmon)),
            (2, 0, Some(Wildlife::Salmon)),
            (-1, 1, Some(Wildlife::Hawk)),
            (-2, 2, Some(Wildlife::Hawk)),
            (-1, 0, Some(Wildlife::Fox)),
            (0, -1, Some(Wildlife::Fox)),
            (1, 1, None),
        ];
        for (q, r, wildlife) in placements {
            let coord = HexCoord::new(q, r);
            board
                .place_tile(coord, any_tile(), 0)
                .expect("test coordinate is empty");
            if let Some(wildlife) = wildlife {
                board
                    .place_wildlife(coord.to_index().unwrap(), wildlife)
                    .expect("test tile accepts wildlife");
            }
        }
        board
    }

    #[test]
    fn hypothetical_wildlife_score_matches_full_existing_and_new_tile_placements() {
        let mut board = representative_board();
        let existing_index = HexCoord::new(1, 1).to_index().unwrap();
        let new_coord = HexCoord::new(2, 1);
        let new_index = new_coord.to_index().unwrap();

        for wildlife in Wildlife::ALL {
            for variant in [
                ScoringCardVariant::A,
                ScoringCardVariant::B,
                ScoringCardVariant::C,
                ScoringCardVariant::D,
            ] {
                for (index, place_tile) in [(existing_index, false), (new_index, true)] {
                    let previous_cell = board.grid.get(index);
                    let previous_positions = board.wildlife_positions[wildlife as usize].clone();
                    let hypothetical =
                        score_wildlife_after_placement(&mut board, wildlife, variant, index);
                    assert_eq!(board.grid.get(index), previous_cell);
                    assert_eq!(
                        board.wildlife_positions[wildlife as usize],
                        previous_positions
                    );

                    let mut full = board.clone();
                    if place_tile {
                        full.place_tile(new_coord, any_tile(), 0)
                            .expect("new test coordinate is empty");
                    }
                    full.place_wildlife(index, wildlife)
                        .expect("full test placement is legal");
                    assert_eq!(
                        hypothetical,
                        score_wildlife(&full, wildlife, variant),
                        "wildlife={wildlife:?}, variant={variant:?}, place_tile={place_tile}"
                    );
                }
            }
        }
    }
}
