use serde::{Deserialize, Serialize};

pub const GRID_RADIUS: i8 = 24;
pub const GRID_DIM: usize = GRID_RADIUS as usize * 2 + 1;
pub const GRID_SIZE: usize = GRID_DIM * GRID_DIM;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct HexCoord {
    pub q: i8,
    pub r: i8,
}

impl HexCoord {
    pub const ORIGIN: Self = Self { q: 0, r: 0 };
    pub const DIRECTIONS: [(i8, i8); 6] = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)];

    pub const fn new(q: i8, r: i8) -> Self {
        Self { q, r }
    }

    pub fn neighbor(self, edge: usize) -> Self {
        let (dq, dr) = Self::DIRECTIONS[edge % 6];
        Self::new(self.q + dq, self.r + dr)
    }

    pub fn neighbors(self) -> [Self; 6] {
        std::array::from_fn(|edge| self.neighbor(edge))
    }

    pub fn to_index(self) -> Option<usize> {
        let q = i16::from(self.q) + i16::from(GRID_RADIUS);
        let r = i16::from(self.r) + i16::from(GRID_RADIUS);
        if !(0..GRID_DIM as i16).contains(&q) || !(0..GRID_DIM as i16).contains(&r) {
            return None;
        }
        Some(q as usize * GRID_DIM + r as usize)
    }

    pub fn from_index(index: usize) -> Option<Self> {
        if index >= GRID_SIZE {
            return None;
        }
        Some(Self::new(
            (index / GRID_DIM) as i8 - GRID_RADIUS,
            (index % GRID_DIM) as i8 - GRID_RADIUS,
        ))
    }

    pub fn distance(self, other: Self) -> u8 {
        let dq = i16::from(self.q) - i16::from(other.q);
        let dr = i16::from(self.r) - i16::from(other.r);
        let ds = -dq - dr;
        ((dq.abs() + dr.abs() + ds.abs()) / 2) as u8
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    proptest! {
        #[test]
        fn coordinate_index_round_trip(q in -24i8..=24, r in -24i8..=24) {
            let coord = HexCoord::new(q, r);
            let round_trip = HexCoord::from_index(coord.to_index().unwrap()).unwrap();
            prop_assert_eq!(round_trip, coord);
        }
    }

    #[test]
    fn grid_supports_the_longest_possible_twenty_turn_chain() {
        assert!(HexCoord::new(21, 0).to_index().is_some());
        assert!(HexCoord::new(-20, 0).to_index().is_some());
    }

    #[test]
    fn neighbor_directions_are_opposites() {
        for edge in 0..6 {
            assert_eq!(
                HexCoord::ORIGIN.neighbor(edge).neighbor((edge + 3) % 6),
                HexCoord::ORIGIN
            );
        }
    }
}
