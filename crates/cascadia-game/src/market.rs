use serde::{Deserialize, Serialize};

use crate::{MarketSlot, Tile, Wildlife};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Market {
    pub tiles: [Option<Tile>; 4],
    pub wildlife: [Option<Wildlife>; 4],
}

impl Market {
    pub const fn empty() -> Self {
        Self {
            tiles: [None; 4],
            wildlife: [None; 4],
        }
    }

    pub fn paired(&self, slot: MarketSlot) -> Option<(Tile, Wildlife)> {
        Some((self.tiles[slot.index()]?, self.wildlife[slot.index()]?))
    }

    pub fn three_of_a_kind(&self) -> Option<Wildlife> {
        Wildlife::ALL.into_iter().find(|wildlife| {
            self.wildlife
                .iter()
                .flatten()
                .filter(|shown| *shown == wildlife)
                .count()
                == 3
        })
    }

    pub fn four_of_a_kind(&self) -> Option<Wildlife> {
        let first = self.wildlife[0]?;
        self.wildlife
            .iter()
            .all(|wildlife| *wildlife == Some(first))
            .then_some(first)
    }

    pub fn wildlife_slots(&self, wildlife: Wildlife) -> Vec<MarketSlot> {
        MarketSlot::ALL
            .into_iter()
            .filter(|slot| self.wildlife[slot.index()] == Some(wildlife))
            .collect()
    }

    pub fn take_paired(&mut self, slot: MarketSlot) -> Option<(Tile, Wildlife)> {
        let tile = self.tiles[slot.index()].take()?;
        let wildlife = self.wildlife[slot.index()].take()?;
        Some((tile, wildlife))
    }

    pub fn take_independent(
        &mut self,
        tile_slot: MarketSlot,
        wildlife_slot: MarketSlot,
    ) -> Option<(Tile, Wildlife)> {
        let tile = self.tiles[tile_slot.index()]?;
        let wildlife = self.wildlife[wildlife_slot.index()]?;
        self.tiles[tile_slot.index()] = None;
        self.wildlife[wildlife_slot.index()] = None;
        Some((tile, wildlife))
    }

    pub fn compact_away_from_draw_stack(&mut self) {
        self.tiles = compact_to_end(self.tiles);
        self.wildlife = compact_to_end(self.wildlife);
    }

    pub fn validate(&self, game_over: bool) -> Result<(), &'static str> {
        if !game_over
            && (self.tiles.iter().any(Option::is_none) || self.wildlife.iter().any(Option::is_none))
        {
            return Err("active game market contains an empty slot");
        }
        if self.four_of_a_kind().is_some() {
            return Err("stable market contains unresolved automatic overpopulation");
        }
        Ok(())
    }
}

impl Default for Market {
    fn default() -> Self {
        Self::empty()
    }
}

fn compact_to_end<T: Copy>(values: [Option<T>; 4]) -> [Option<T>; 4] {
    let present: Vec<T> = values.into_iter().flatten().collect();
    let mut compacted = [None; 4];
    let offset = 4 - present.len();
    for (index, value) in present.into_iter().enumerate() {
        compacted[offset + index] = Some(value);
    }
    compacted
}

#[cfg(test)]
mod tests {
    use crate::STANDARD_TILES;

    use super::*;

    #[test]
    fn independent_draft_leaves_unchosen_components_in_place() {
        let mut market = Market {
            tiles: [
                Some(STANDARD_TILES[0]),
                Some(STANDARD_TILES[1]),
                Some(STANDARD_TILES[2]),
                Some(STANDARD_TILES[3]),
            ],
            wildlife: [
                Some(Wildlife::Bear),
                Some(Wildlife::Elk),
                Some(Wildlife::Salmon),
                Some(Wildlife::Hawk),
            ],
        };

        let drafted = market
            .take_independent(MarketSlot::ZERO, MarketSlot::ONE)
            .unwrap();
        assert_eq!(drafted, (STANDARD_TILES[0], Wildlife::Elk));
        assert_eq!(market.wildlife[0], Some(Wildlife::Bear));
        assert_eq!(market.tiles[1], Some(STANDARD_TILES[1]));
        assert!(market.tiles[0].is_none());
        assert!(market.wildlife[1].is_none());
    }

    #[test]
    fn independent_draft_allows_the_same_slot_and_removes_both_components() {
        let mut market = Market {
            tiles: [
                Some(STANDARD_TILES[0]),
                Some(STANDARD_TILES[1]),
                Some(STANDARD_TILES[2]),
                Some(STANDARD_TILES[3]),
            ],
            wildlife: [
                Some(Wildlife::Bear),
                Some(Wildlife::Elk),
                Some(Wildlife::Salmon),
                Some(Wildlife::Hawk),
            ],
        };

        assert_eq!(
            market.take_independent(MarketSlot::THREE, MarketSlot::THREE),
            Some((STANDARD_TILES[3], Wildlife::Hawk))
        );
        assert_eq!(market.tiles[3], None);
        assert_eq!(market.wildlife[3], None);
    }

    #[test]
    fn solo_compaction_preserves_relative_order() {
        let mut market = Market {
            tiles: [Some(STANDARD_TILES[0]), None, Some(STANDARD_TILES[2]), None],
            wildlife: [None, Some(Wildlife::Elk), None, Some(Wildlife::Hawk)],
        };
        market.compact_away_from_draw_stack();

        assert_eq!(
            market.tiles,
            [None, None, Some(STANDARD_TILES[0]), Some(STANDARD_TILES[2])]
        );
        assert_eq!(
            market.wildlife,
            [None, None, Some(Wildlife::Elk), Some(Wildlife::Hawk)]
        );
    }
}
