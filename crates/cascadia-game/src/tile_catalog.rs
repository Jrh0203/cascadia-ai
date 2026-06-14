use serde::{Deserialize, Serialize};

use crate::{HexCoord, Rotation, Terrain, Tile, Wildlife};

const BEAR: u8 = 1 << Wildlife::Bear as u8;
const ELK: u8 = 1 << Wildlife::Elk as u8;
const SALMON: u8 = 1 << Wildlife::Salmon as u8;
const HAWK: u8 = 1 << Wildlife::Hawk as u8;
const FOX: u8 = 1 << Wildlife::Fox as u8;

/// The 85 standard habitat tiles.
///
/// IDs are stable serialization identifiers, not the printed component numbers.
pub const STANDARD_TILES: [Tile; 85] = [
    // Mountain keystones.
    Tile::keystone(0, Terrain::Mountain, Wildlife::Hawk),
    Tile::keystone(1, Terrain::Mountain, Wildlife::Hawk),
    Tile::keystone(2, Terrain::Mountain, Wildlife::Bear),
    Tile::keystone(3, Terrain::Mountain, Wildlife::Elk),
    Tile::keystone(4, Terrain::Mountain, Wildlife::Elk),
    // Forest keystones.
    Tile::keystone(5, Terrain::Forest, Wildlife::Bear),
    Tile::keystone(6, Terrain::Forest, Wildlife::Bear),
    Tile::keystone(7, Terrain::Forest, Wildlife::Elk),
    Tile::keystone(8, Terrain::Forest, Wildlife::Fox),
    Tile::keystone(9, Terrain::Forest, Wildlife::Fox),
    // Prairie keystones.
    Tile::keystone(10, Terrain::Prairie, Wildlife::Elk),
    Tile::keystone(11, Terrain::Prairie, Wildlife::Elk),
    Tile::keystone(12, Terrain::Prairie, Wildlife::Fox),
    Tile::keystone(13, Terrain::Prairie, Wildlife::Salmon),
    Tile::keystone(14, Terrain::Prairie, Wildlife::Salmon),
    // Wetland keystones.
    Tile::keystone(15, Terrain::Wetland, Wildlife::Fox),
    Tile::keystone(16, Terrain::Wetland, Wildlife::Fox),
    Tile::keystone(17, Terrain::Wetland, Wildlife::Salmon),
    Tile::keystone(18, Terrain::Wetland, Wildlife::Salmon),
    Tile::keystone(19, Terrain::Wetland, Wildlife::Hawk),
    // River keystones.
    Tile::keystone(20, Terrain::River, Wildlife::Hawk),
    Tile::keystone(21, Terrain::River, Wildlife::Hawk),
    Tile::keystone(22, Terrain::River, Wildlife::Salmon),
    Tile::keystone(23, Terrain::River, Wildlife::Bear),
    Tile::keystone(24, Terrain::River, Wildlife::Bear),
    // Mountain / Forest.
    Tile::dual(25, Terrain::Mountain, Terrain::Forest, HAWK | BEAR | ELK),
    Tile::dual(26, Terrain::Mountain, Terrain::Forest, FOX | BEAR | ELK),
    Tile::dual(27, Terrain::Mountain, Terrain::Forest, HAWK | BEAR),
    Tile::dual(28, Terrain::Mountain, Terrain::Forest, HAWK | ELK),
    Tile::dual(29, Terrain::Mountain, Terrain::Forest, BEAR | FOX),
    Tile::dual(30, Terrain::Mountain, Terrain::Forest, ELK | FOX),
    // Mountain / Prairie.
    Tile::dual(31, Terrain::Mountain, Terrain::Prairie, FOX | BEAR | ELK),
    Tile::dual(32, Terrain::Mountain, Terrain::Prairie, SALMON | FOX | BEAR),
    Tile::dual(33, Terrain::Mountain, Terrain::Prairie, HAWK | ELK),
    Tile::dual(34, Terrain::Mountain, Terrain::Prairie, HAWK | FOX),
    Tile::dual(35, Terrain::Mountain, Terrain::Prairie, BEAR | SALMON),
    Tile::dual(36, Terrain::Mountain, Terrain::Prairie, ELK | SALMON),
    // Mountain / Wetland.
    Tile::dual(37, Terrain::Mountain, Terrain::Wetland, FOX | HAWK | BEAR),
    Tile::dual(38, Terrain::Mountain, Terrain::Wetland, SALMON | BEAR | ELK),
    Tile::dual(39, Terrain::Mountain, Terrain::Wetland, HAWK | SALMON),
    Tile::dual(40, Terrain::Mountain, Terrain::Wetland, BEAR | SALMON),
    Tile::dual(41, Terrain::Mountain, Terrain::Wetland, ELK | FOX),
    Tile::dual(42, Terrain::Mountain, Terrain::Wetland, ELK | HAWK),
    // Forest / Prairie.
    Tile::dual(43, Terrain::Forest, Terrain::Prairie, SALMON | FOX | ELK),
    Tile::dual(44, Terrain::Forest, Terrain::Prairie, BEAR | ELK),
    Tile::dual(45, Terrain::Forest, Terrain::Prairie, BEAR | FOX),
    Tile::dual(46, Terrain::Forest, Terrain::Prairie, ELK | FOX),
    Tile::dual(47, Terrain::Forest, Terrain::Prairie, ELK | SALMON),
    Tile::dual(48, Terrain::Forest, Terrain::Prairie, FOX | SALMON),
    // Forest / Wetland.
    Tile::dual(49, Terrain::Forest, Terrain::Wetland, SALMON | HAWK | ELK),
    Tile::dual(50, Terrain::Forest, Terrain::Wetland, BEAR | FOX),
    Tile::dual(51, Terrain::Forest, Terrain::Wetland, BEAR | SALMON),
    Tile::dual(52, Terrain::Forest, Terrain::Wetland, ELK | SALMON),
    Tile::dual(53, Terrain::Forest, Terrain::Wetland, ELK | HAWK),
    Tile::dual(54, Terrain::Forest, Terrain::Wetland, FOX | HAWK),
    // Forest / River.
    Tile::dual(55, Terrain::Forest, Terrain::River, HAWK | FOX | ELK),
    Tile::dual(56, Terrain::Forest, Terrain::River, BEAR | SALMON),
    Tile::dual(57, Terrain::Forest, Terrain::River, FOX | SALMON),
    Tile::dual(58, Terrain::Forest, Terrain::River, ELK | HAWK),
    Tile::dual(59, Terrain::Forest, Terrain::River, ELK | BEAR),
    Tile::dual(60, Terrain::Forest, Terrain::River, FOX | BEAR),
    // Prairie / Wetland.
    Tile::dual(61, Terrain::Prairie, Terrain::Wetland, SALMON | FOX | ELK),
    Tile::dual(62, Terrain::Prairie, Terrain::Wetland, SALMON | FOX | HAWK),
    Tile::dual(63, Terrain::Prairie, Terrain::Wetland, ELK | FOX),
    Tile::dual(64, Terrain::Prairie, Terrain::Wetland, ELK | SALMON),
    Tile::dual(65, Terrain::Prairie, Terrain::Wetland, FOX | HAWK),
    Tile::dual(66, Terrain::Prairie, Terrain::Wetland, SALMON | HAWK),
    // Prairie / River.
    Tile::dual(67, Terrain::Prairie, Terrain::River, SALMON | FOX | BEAR),
    Tile::dual(68, Terrain::Prairie, Terrain::River, FOX | HAWK | BEAR),
    Tile::dual(69, Terrain::Prairie, Terrain::River, ELK | SALMON),
    Tile::dual(70, Terrain::Prairie, Terrain::River, ELK | HAWK),
    Tile::dual(71, Terrain::Prairie, Terrain::River, FOX | HAWK),
    Tile::dual(72, Terrain::Prairie, Terrain::River, FOX | BEAR),
    // Wetland / River.
    Tile::dual(73, Terrain::Wetland, Terrain::River, SALMON | HAWK | BEAR),
    Tile::dual(74, Terrain::Wetland, Terrain::River, FOX | SALMON),
    Tile::dual(75, Terrain::Wetland, Terrain::River, FOX | HAWK),
    Tile::dual(76, Terrain::Wetland, Terrain::River, SALMON | HAWK),
    Tile::dual(77, Terrain::Wetland, Terrain::River, SALMON | BEAR),
    Tile::dual(78, Terrain::Wetland, Terrain::River, HAWK | BEAR),
    // River / Mountain.
    Tile::dual(79, Terrain::River, Terrain::Mountain, SALMON | HAWK | BEAR),
    Tile::dual(80, Terrain::River, Terrain::Mountain, SALMON | HAWK),
    Tile::dual(81, Terrain::River, Terrain::Mountain, SALMON | BEAR),
    Tile::dual(82, Terrain::River, Terrain::Mountain, HAWK | BEAR),
    Tile::dual(83, Terrain::River, Terrain::Mountain, HAWK | ELK),
    Tile::dual(84, Terrain::River, Terrain::Mountain, BEAR | ELK),
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct StarterPlacement {
    pub coord: HexCoord,
    pub tile: Tile,
    pub rotation: Rotation,
}

const P0: HexCoord = HexCoord::new(0, 0);
const P1: HexCoord = HexCoord::new(0, 1);
const P2: HexCoord = HexCoord::new(1, 0);

pub const STARTER_CLUSTERS: [[StarterPlacement; 3]; 5] = [
    [
        StarterPlacement {
            coord: P0,
            tile: Tile::keystone(100, Terrain::Mountain, Wildlife::Bear),
            rotation: Rotation::ZERO,
        },
        StarterPlacement {
            coord: P1,
            tile: Tile::dual(101, Terrain::Forest, Terrain::Wetland, HAWK | FOX | ELK),
            rotation: Rotation::FIVE,
        },
        StarterPlacement {
            coord: P2,
            tile: Tile::dual(102, Terrain::Prairie, Terrain::River, SALMON | BEAR),
            rotation: Rotation::FOUR,
        },
    ],
    [
        StarterPlacement {
            coord: P0,
            tile: Tile::keystone(103, Terrain::Wetland, Wildlife::Hawk),
            rotation: Rotation::ZERO,
        },
        StarterPlacement {
            coord: P1,
            tile: Tile::dual(104, Terrain::Forest, Terrain::River, SALMON | HAWK | ELK),
            rotation: Rotation::FIVE,
        },
        StarterPlacement {
            coord: P2,
            tile: Tile::dual(105, Terrain::Mountain, Terrain::Prairie, BEAR | FOX),
            rotation: Rotation::FOUR,
        },
    ],
    [
        StarterPlacement {
            coord: P0,
            tile: Tile::keystone(106, Terrain::Prairie, Wildlife::Fox),
            rotation: Rotation::ZERO,
        },
        StarterPlacement {
            coord: P1,
            tile: Tile::dual(107, Terrain::Wetland, Terrain::River, SALMON | FOX | HAWK),
            rotation: Rotation::FIVE,
        },
        StarterPlacement {
            coord: P2,
            tile: Tile::dual(108, Terrain::Mountain, Terrain::Forest, BEAR | ELK),
            rotation: Rotation::FOUR,
        },
    ],
    [
        StarterPlacement {
            coord: P0,
            tile: Tile::keystone(109, Terrain::Forest, Wildlife::Elk),
            rotation: Rotation::ZERO,
        },
        StarterPlacement {
            coord: P1,
            tile: Tile::dual(110, Terrain::River, Terrain::Mountain, HAWK | BEAR | ELK),
            rotation: Rotation::FIVE,
        },
        StarterPlacement {
            coord: P2,
            tile: Tile::dual(111, Terrain::Prairie, Terrain::Wetland, FOX | SALMON),
            rotation: Rotation::FOUR,
        },
    ],
    [
        StarterPlacement {
            coord: P0,
            tile: Tile::keystone(112, Terrain::River, Wildlife::Salmon),
            rotation: Rotation::ZERO,
        },
        StarterPlacement {
            coord: P1,
            tile: Tile::dual(113, Terrain::Forest, Terrain::Prairie, SALMON | BEAR | ELK),
            rotation: Rotation::FIVE,
        },
        StarterPlacement {
            coord: P2,
            tile: Tile::dual(114, Terrain::Mountain, Terrain::Wetland, FOX | HAWK),
            rotation: Rotation::FOUR,
        },
    ],
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_catalog_matches_component_invariants() {
        assert_eq!(STANDARD_TILES.len(), 85);
        assert_eq!(
            STANDARD_TILES.iter().filter(|tile| tile.keystone).count(),
            25
        );

        for terrain in Terrain::ALL {
            assert_eq!(
                STANDARD_TILES
                    .iter()
                    .filter(|tile| tile.contains_terrain(terrain))
                    .count(),
                29,
                "{terrain:?}"
            );
        }

        for wildlife in Wildlife::ALL {
            assert_eq!(
                STANDARD_TILES
                    .iter()
                    .filter(|tile| tile.wildlife.contains(wildlife))
                    .count(),
                32,
                "{wildlife:?}"
            );
        }
    }

    #[test]
    fn every_unordered_terrain_pair_occurs_six_times() {
        for (left_index, left) in Terrain::ALL.into_iter().enumerate() {
            for right in Terrain::ALL.into_iter().skip(left_index + 1) {
                let count = STANDARD_TILES
                    .iter()
                    .filter(|tile| tile.contains_terrain(left) && tile.contains_terrain(right))
                    .count();
                assert_eq!(count, 6, "{left:?}/{right:?}");
            }
        }
    }

    #[test]
    fn tile_ids_are_unique() {
        let mut ids: Vec<_> = STANDARD_TILES.iter().map(|tile| tile.id.0).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), STANDARD_TILES.len());
    }
}
