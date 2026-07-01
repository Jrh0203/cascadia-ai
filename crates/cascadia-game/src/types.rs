use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum Terrain {
    Mountain = 0,
    Forest = 1,
    Prairie = 2,
    Wetland = 3,
    River = 4,
}

impl Terrain {
    pub const ALL: [Self; 5] = [
        Self::Mountain,
        Self::Forest,
        Self::Prairie,
        Self::Wetland,
        Self::River,
    ];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum Wildlife {
    Bear = 0,
    Elk = 1,
    Salmon = 2,
    Hawk = 3,
    Fox = 4,
}

impl Wildlife {
    pub const ALL: [Self; 5] = [Self::Bear, Self::Elk, Self::Salmon, Self::Hawk, Self::Fox];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct WildlifeMask(u8);

impl WildlifeMask {
    pub const EMPTY: Self = Self(0);

    pub const fn from_bits(bits: u8) -> Self {
        Self(bits & 0b1_1111)
    }

    pub const fn one(wildlife: Wildlife) -> Self {
        Self(1 << wildlife as u8)
    }

    pub const fn contains(self, wildlife: Wildlife) -> bool {
        self.0 & (1 << wildlife as u8) != 0
    }

    pub const fn bits(self) -> u8 {
        self.0
    }

    pub fn iter(self) -> impl Iterator<Item = Wildlife> {
        Wildlife::ALL
            .into_iter()
            .filter(move |wildlife| self.contains(*wildlife))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TileId(pub u8);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Tile {
    pub id: TileId,
    pub terrain_a: Terrain,
    pub terrain_b: Option<Terrain>,
    pub wildlife: WildlifeMask,
    pub keystone: bool,
}

impl Tile {
    pub const fn keystone(id: u8, terrain: Terrain, wildlife: Wildlife) -> Self {
        Self {
            id: TileId(id),
            terrain_a: terrain,
            terrain_b: None,
            wildlife: WildlifeMask::one(wildlife),
            keystone: true,
        }
    }

    pub const fn dual(id: u8, terrain_a: Terrain, terrain_b: Terrain, wildlife_bits: u8) -> Self {
        Self {
            id: TileId(id),
            terrain_a,
            terrain_b: Some(terrain_b),
            wildlife: WildlifeMask::from_bits(wildlife_bits),
            keystone: false,
        }
    }

    pub const fn contains_terrain(self, terrain: Terrain) -> bool {
        self.terrain_a as u8 == terrain as u8
            || match self.terrain_b {
                Some(other) => other as u8 == terrain as u8,
                None => false,
            }
    }

    pub fn terrain_on_edge(self, rotation: Rotation, edge: usize) -> Terrain {
        let Some(terrain_b) = self.terrain_b else {
            return self.terrain_a;
        };
        let offset = (edge + 6 - usize::from(rotation.get())) % 6;
        if offset < 3 {
            self.terrain_a
        } else {
            terrain_b
        }
    }

    pub const fn canonical_rotation(self, rotation: Rotation) -> Rotation {
        if self.terrain_b.is_some() {
            rotation
        } else {
            Rotation::ZERO
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Rotation(u8);

impl Rotation {
    pub const ZERO: Self = Self(0);
    pub const ONE: Self = Self(1);
    pub const TWO: Self = Self(2);
    pub const THREE: Self = Self(3);
    pub const FOUR: Self = Self(4);
    pub const FIVE: Self = Self(5);
    pub const ALL: [Self; 6] = [
        Self::ZERO,
        Self::ONE,
        Self::TWO,
        Self::THREE,
        Self::FOUR,
        Self::FIVE,
    ];

    pub const fn new(value: u8) -> Option<Self> {
        if value < 6 { Some(Self(value)) } else { None }
    }

    pub const fn get(self) -> u8 {
        self.0
    }
}

impl Default for Rotation {
    fn default() -> Self {
        Self::ZERO
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct MarketSlot(u8);

impl MarketSlot {
    pub const ZERO: Self = Self(0);
    pub const ONE: Self = Self(1);
    pub const TWO: Self = Self(2);
    pub const THREE: Self = Self(3);
    pub const ALL: [Self; 4] = [Self::ZERO, Self::ONE, Self::TWO, Self::THREE];

    pub const fn new(value: u8) -> Option<Self> {
        if value < 4 { Some(Self(value)) } else { None }
    }

    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ScoringVariant {
    A,
    B,
    C,
    D,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ScoringCards {
    pub bear: ScoringVariant,
    pub elk: ScoringVariant,
    pub salmon: ScoringVariant,
    pub hawk: ScoringVariant,
    pub fox: ScoringVariant,
}

impl ScoringCards {
    pub const AAAAA: Self = Self {
        bear: ScoringVariant::A,
        elk: ScoringVariant::A,
        salmon: ScoringVariant::A,
        hawk: ScoringVariant::A,
        fox: ScoringVariant::A,
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dual_tile_edges_rotate_without_changing_composition() {
        let tile = Tile::dual(
            0,
            Terrain::Forest,
            Terrain::River,
            (1 << Wildlife::Bear as u8) | (1 << Wildlife::Hawk as u8),
        );

        for rotation in 0..6 {
            let rotation = Rotation::new(rotation).unwrap();
            let forest_edges = (0..6)
                .filter(|edge| tile.terrain_on_edge(rotation, *edge) == Terrain::Forest)
                .count();
            assert_eq!(forest_edges, 3);
        }
    }

    #[test]
    fn bounded_values_reject_invalid_input() {
        assert!(Rotation::new(6).is_none());
        assert!(MarketSlot::new(4).is_none());
    }
}
