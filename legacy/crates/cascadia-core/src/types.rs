/// Terrain types on hex tiles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Terrain {
    Forest = 0,
    Prairie = 1,
    Wetland = 2,
    Mountain = 3,
    River = 4,
}

impl Terrain {
    pub const ALL: [Terrain; 5] = [
        Terrain::Forest,
        Terrain::Prairie,
        Terrain::Wetland,
        Terrain::Mountain,
        Terrain::River,
    ];

    #[inline(always)]
    pub fn from_u8(v: u8) -> Option<Terrain> {
        match v {
            0 => Some(Terrain::Forest),
            1 => Some(Terrain::Prairie),
            2 => Some(Terrain::Wetland),
            3 => Some(Terrain::Mountain),
            4 => Some(Terrain::River),
            _ => None,
        }
    }
}

/// Wildlife token types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Wildlife {
    Bear = 0,
    Elk = 1,
    Salmon = 2,
    Hawk = 3,
    Fox = 4,
}

impl Wildlife {
    pub const ALL: [Wildlife; 5] = [
        Wildlife::Bear,
        Wildlife::Elk,
        Wildlife::Salmon,
        Wildlife::Hawk,
        Wildlife::Fox,
    ];

    pub const COUNT: usize = 5;

    #[inline(always)]
    pub fn from_u8(v: u8) -> Option<Wildlife> {
        match v {
            0 => Some(Wildlife::Bear),
            1 => Some(Wildlife::Elk),
            2 => Some(Wildlife::Salmon),
            3 => Some(Wildlife::Hawk),
            4 => Some(Wildlife::Fox),
            _ => None,
        }
    }

    pub fn emoji(self) -> &'static str {
        match self {
            Wildlife::Bear => "🐻",
            Wildlife::Elk => "🫎",
            Wildlife::Salmon => "🐟",
            Wildlife::Hawk => "🦅",
            Wildlife::Fox => "🦊",
        }
    }
}

/// A bitmask of allowed wildlife types on a tile (5 bits).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WildlifeMask(pub u8);

impl WildlifeMask {
    pub const NONE: WildlifeMask = WildlifeMask(0);

    #[inline(always)]
    pub fn new(wildlife: &[Wildlife]) -> Self {
        let mut mask = 0u8;
        for &w in wildlife {
            mask |= 1 << (w as u8);
        }
        WildlifeMask(mask)
    }

    #[inline(always)]
    pub fn contains(self, w: Wildlife) -> bool {
        (self.0 >> (w as u8)) & 1 != 0
    }

    #[inline(always)]
    pub fn count(self) -> u32 {
        self.0.count_ones()
    }

    pub fn iter(self) -> impl Iterator<Item = Wildlife> {
        Wildlife::ALL.into_iter().filter(move |&w| self.contains(w))
    }
}

/// Packed cell representation in a u16 bitfield.
///
/// Layout (low to high):
///   bits  0-2:  primary terrain (0-4 valid, 7 = empty)
///   bits  3-5:  secondary terrain (0-4 valid, 7 = single-terrain)
///   bits  6-8:  placed wildlife (0-4 valid, 7 = none placed)
///   bits  9-13: allowed wildlife mask (5 bits)
///   bit   14:   tile present
///   bit   15:   keystone tile
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct Cell(pub u16);

impl Cell {
    pub const EMPTY: Cell = Cell(0b0_0_00000_111_111_111);

    const TERRAIN1_MASK: u16 = 0b111;
    const TERRAIN2_SHIFT: u32 = 3;
    const TERRAIN2_MASK: u16 = 0b111 << 3;
    const WILDLIFE_SHIFT: u32 = 6;
    const WILDLIFE_MASK: u16 = 0b111 << 6;
    const ALLOWED_SHIFT: u32 = 9;
    const ALLOWED_MASK: u16 = 0b11111 << 9;
    const PRESENT_BIT: u16 = 1 << 14;
    const KEYSTONE_BIT: u16 = 1 << 15;

    #[inline(always)]
    pub fn new(
        terrain1: Terrain,
        terrain2: Option<Terrain>,
        allowed: WildlifeMask,
        keystone: bool,
    ) -> Self {
        let t1 = terrain1 as u16;
        let t2 = terrain2.map_or(7u16, |t| t as u16);
        let bits = t1
            | (t2 << Self::TERRAIN2_SHIFT)
            | (7u16 << Self::WILDLIFE_SHIFT) // no wildlife placed
            | ((allowed.0 as u16) << Self::ALLOWED_SHIFT)
            | Self::PRESENT_BIT
            | if keystone { Self::KEYSTONE_BIT } else { 0 };
        Cell(bits)
    }

    #[inline(always)]
    pub fn is_present(self) -> bool {
        self.0 & Self::PRESENT_BIT != 0
    }

    #[inline(always)]
    pub fn is_keystone(self) -> bool {
        self.0 & Self::KEYSTONE_BIT != 0
    }

    #[inline(always)]
    pub fn primary_terrain(self) -> Option<Terrain> {
        if !self.is_present() {
            return None;
        }
        Terrain::from_u8((self.0 & Self::TERRAIN1_MASK) as u8)
    }

    #[inline(always)]
    pub fn secondary_terrain(self) -> Option<Terrain> {
        if !self.is_present() {
            return None;
        }
        let v = ((self.0 & Self::TERRAIN2_MASK) >> Self::TERRAIN2_SHIFT) as u8;
        Terrain::from_u8(v)
    }

    /// Returns both terrains. For single-terrain tiles, second is None.
    #[inline(always)]
    pub fn terrains(self) -> (Option<Terrain>, Option<Terrain>) {
        (self.primary_terrain(), self.secondary_terrain())
    }

    #[inline(always)]
    pub fn placed_wildlife(self) -> Option<Wildlife> {
        let v = ((self.0 & Self::WILDLIFE_MASK) >> Self::WILDLIFE_SHIFT) as u8;
        Wildlife::from_u8(v)
    }

    #[inline(always)]
    pub fn has_wildlife(self) -> bool {
        ((self.0 & Self::WILDLIFE_MASK) >> Self::WILDLIFE_SHIFT) < 5
    }

    #[inline(always)]
    pub fn allowed_wildlife(self) -> WildlifeMask {
        WildlifeMask(((self.0 & Self::ALLOWED_MASK) >> Self::ALLOWED_SHIFT) as u8)
    }

    #[inline(always)]
    pub fn can_place_wildlife(self, w: Wildlife) -> bool {
        self.is_present() && !self.has_wildlife() && self.allowed_wildlife().contains(w)
    }

    /// Place a wildlife token on this cell. Returns the updated cell.
    #[inline(always)]
    pub fn with_wildlife(self, w: Wildlife) -> Cell {
        let cleared = self.0 & !Self::WILDLIFE_MASK;
        Cell(cleared | ((w as u16) << Self::WILDLIFE_SHIFT))
    }

    /// Remove placed wildlife. Returns the updated cell.
    #[inline(always)]
    pub fn without_wildlife(self) -> Cell {
        Cell(self.0 | Self::WILDLIFE_MASK) // set to 0b111 = no wildlife
    }
}

impl std::fmt::Debug for Cell {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if !self.is_present() {
            write!(f, "Cell(empty)")
        } else {
            write!(
                f,
                "Cell({:?}/{:?} w={:?} allowed={:05b} ks={})",
                self.primary_terrain(),
                self.secondary_terrain(),
                self.placed_wildlife(),
                self.allowed_wildlife().0,
                self.is_keystone()
            )
        }
    }
}

/// Data needed to create a tile (before placement on the board).
#[derive(Debug, Clone, Copy)]
pub struct TileData {
    pub terrain1: Terrain,
    pub terrain2: Option<Terrain>,
    pub allowed: WildlifeMask,
    pub keystone: bool,
}

impl TileData {
    pub fn single(terrain: Terrain, allowed: WildlifeMask) -> Self {
        TileData {
            terrain1: terrain,
            terrain2: None,
            allowed,
            keystone: true,
        }
    }

    pub fn dual(terrain1: Terrain, terrain2: Terrain, allowed: WildlifeMask) -> Self {
        TileData {
            terrain1,
            terrain2: Some(terrain2),
            allowed,
            keystone: false,
        }
    }

    pub fn to_cell(self) -> Cell {
        Cell::new(self.terrain1, self.terrain2, self.allowed, self.keystone)
    }
}

/// Which scoring card variant is in use for each wildlife type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ScoringCardVariant {
    A = 0,
    B = 1,
    C = 2,
    D = 3,
}

/// The set of scoring cards for a game (one variant per wildlife type).
#[derive(Debug, Clone, Copy)]
pub struct ScoringCards {
    pub cards: [ScoringCardVariant; 5],
}

impl ScoringCards {
    pub fn all_a() -> Self {
        ScoringCards {
            cards: [ScoringCardVariant::A; 5],
        }
    }

    pub fn variant_for(&self, w: Wildlife) -> ScoringCardVariant {
        self.cards[w as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cell_empty() {
        let c = Cell::EMPTY;
        assert!(!c.is_present());
        assert!(!c.has_wildlife());
        assert_eq!(c.primary_terrain(), None);
    }

    #[test]
    fn cell_roundtrip() {
        let mask = WildlifeMask::new(&[Wildlife::Bear, Wildlife::Salmon]);
        let c = Cell::new(Terrain::Forest, Some(Terrain::River), mask, false);

        assert!(c.is_present());
        assert!(!c.is_keystone());
        assert_eq!(c.primary_terrain(), Some(Terrain::Forest));
        assert_eq!(c.secondary_terrain(), Some(Terrain::River));
        assert!(!c.has_wildlife());
        assert!(c.allowed_wildlife().contains(Wildlife::Bear));
        assert!(c.allowed_wildlife().contains(Wildlife::Salmon));
        assert!(!c.allowed_wildlife().contains(Wildlife::Elk));
    }

    #[test]
    fn cell_wildlife_placement() {
        let mask = WildlifeMask::new(&[Wildlife::Bear, Wildlife::Elk]);
        let c = Cell::new(Terrain::Mountain, None, mask, true);

        assert!(c.is_keystone());
        assert!(c.can_place_wildlife(Wildlife::Bear));
        assert!(!c.can_place_wildlife(Wildlife::Fox));

        let c2 = c.with_wildlife(Wildlife::Bear);
        assert!(c2.has_wildlife());
        assert_eq!(c2.placed_wildlife(), Some(Wildlife::Bear));
        assert!(!c2.can_place_wildlife(Wildlife::Elk)); // already has wildlife

        let c3 = c2.without_wildlife();
        assert!(!c3.has_wildlife());
        assert!(c3.can_place_wildlife(Wildlife::Elk));
    }

    #[test]
    fn wildlife_mask_iter() {
        let mask = WildlifeMask::new(&[Wildlife::Hawk, Wildlife::Fox]);
        let v: Vec<_> = mask.iter().collect();
        assert_eq!(v, vec![Wildlife::Hawk, Wildlife::Fox]);
    }
}
