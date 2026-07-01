use std::sync::OnceLock;

use cascadia_game::{
    D6Transform, GameMode, GameState, Market, MarketSlot, PublicGameState, PublicSupply, Rotation,
    STANDARD_TILES, Terrain, Tile, Wildlife, WildlifeMask,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const EXACT_SEMANTIC_SUPPLY_SCHEMA_VERSION: u16 = 1;
pub const EXACT_SEMANTIC_SUPPLY_SCHEMA: &str = "exact-semantic-supply-v1";
pub const CANONICAL_TILE_ARCHETYPE_SCHEMA: &str = "canonical-public-tile-archetype-v1";
pub const MAX_EXACT_REFILL_SLOTS: u8 = 4;

const ARCHETYPE_BYTES: usize = 10;
const CATALOG_MAGIC: &[u8; 8] = b"CSSCAT1\0";
const SUPPLY_MAGIC: &[u8; 8] = b"CSSSUP1\0";
const REFILL_MAGIC: &[u8; 8] = b"CSSRFL1\0";
const ABSENT_TERRAIN: u8 = u8::MAX;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SemanticArchetypeId(u16);

impl SemanticArchetypeId {
    pub const fn index(self) -> usize {
        self.0 as usize
    }

    pub const fn code(self) -> u16 {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CanonicalTileArchetype {
    pub primary_terrain: Terrain,
    pub secondary_terrain: Option<Terrain>,
    pub directed_edges: [Terrain; 6],
    pub wildlife: WildlifeMask,
    pub keystone: bool,
}

impl CanonicalTileArchetype {
    pub fn from_tile(tile: Tile) -> Self {
        let (primary_terrain, secondary_terrain) = canonical_terrains(tile);
        let directed_edges = canonical_edge_ring(tile);
        Self {
            primary_terrain,
            secondary_terrain,
            directed_edges,
            wildlife: tile.wildlife,
            keystone: tile.keystone,
        }
    }

    pub fn terrain_on_edge(self, rotation: Rotation, edge: usize) -> Terrain {
        let offset = (edge + 6 - usize::from(rotation.get())) % 6;
        self.directed_edges[offset]
    }

    pub fn contains_terrain(self, terrain: Terrain) -> bool {
        self.primary_terrain == terrain || self.secondary_terrain == Some(terrain)
    }

    pub fn canonical_bytes(self) -> [u8; ARCHETYPE_BYTES] {
        let mut bytes = [0u8; ARCHETYPE_BYTES];
        bytes[0] = self.primary_terrain as u8;
        bytes[1] = self
            .secondary_terrain
            .map_or(ABSENT_TERRAIN, |terrain| terrain as u8);
        for (index, terrain) in self.directed_edges.into_iter().enumerate() {
            bytes[2 + index] = terrain as u8;
        }
        bytes[8] = self.wildlife.bits();
        bytes[9] = u8::from(self.keystone);
        bytes
    }

    pub fn canonical_hash(self) -> blake3::Hash {
        blake3::hash(&self.canonical_bytes())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SemanticTileReference {
    pub archetype_id: SemanticArchetypeId,
    pub tile_to_canonical_rotation: Rotation,
}

impl SemanticTileReference {
    pub fn canonical_rotation_for_game(self, game_rotation: Rotation) -> Rotation {
        Rotation::ALL
            [usize::from((game_rotation.get() + 6 - self.tile_to_canonical_rotation.get()) % 6)]
    }

    pub fn game_rotation_for_canonical(self, canonical_rotation: Rotation) -> Rotation {
        Rotation::ALL
            [usize::from((canonical_rotation.get() + self.tile_to_canonical_rotation.get()) % 6)]
    }

    pub fn frontier_compatibility(
        self,
        tile: Tile,
        requirements: FrontierTerrainRequirements,
    ) -> Result<FrontierCompatibility, SemanticSupplyError> {
        let expected = standard_semantic_archetype_catalog().reference_for_tile(tile)?;
        if expected != self {
            return Err(SemanticSupplyError::TileReferenceMismatch);
        }
        Ok(frontier_compatibility(tile, requirements))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SemanticMarketLink {
    pub slot: MarketSlot,
    pub tile: SemanticTileReference,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrontierTerrainRequirements {
    /// At index `e`, the public terrain on the adjacent tile's edge facing
    /// this placement. `None` means that edge has no neighboring tile.
    pub neighbor_facing_terrains: [Option<Terrain>; 6],
}

impl FrontierTerrainRequirements {
    pub const fn new(neighbor_facing_terrains: [Option<Terrain>; 6]) -> Self {
        Self {
            neighbor_facing_terrains,
        }
    }

    pub fn present_edges(self) -> u8 {
        self.neighbor_facing_terrains
            .into_iter()
            .filter(Option::is_some)
            .count() as u8
    }

    pub fn transformed(self, transform: D6Transform) -> Result<Self, SemanticSupplyError> {
        let mut transformed = [None; 6];
        for (edge, terrain) in self.neighbor_facing_terrains.into_iter().enumerate() {
            transformed[transform.transform_edge(edge)?] = terrain;
        }
        Ok(Self::new(transformed))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrontierCompatibility {
    pub matching_edges_by_rotation: [u8; 6],
    pub all_present_match_rotation_mask: u8,
    pub best_matching_edges: u8,
}

impl FrontierCompatibility {
    pub fn matching_edges(self, rotation: Rotation) -> u8 {
        self.matching_edges_by_rotation[usize::from(rotation.get())]
    }

    pub fn all_present_edges_match(self, rotation: Rotation) -> bool {
        self.all_present_match_rotation_mask & (1 << rotation.get()) != 0
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SemanticArchetypeDefinition {
    pub id: SemanticArchetypeId,
    pub archetype: CanonicalTileArchetype,
    pub standard_tile_count: u16,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemanticArchetypeCatalog {
    definitions: Vec<SemanticArchetypeDefinition>,
    canonical_blake3: [u8; 32],
}

impl SemanticArchetypeCatalog {
    pub fn definitions(&self) -> &[SemanticArchetypeDefinition] {
        &self.definitions
    }

    pub fn len(&self) -> usize {
        self.definitions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.definitions.is_empty()
    }

    pub fn canonical_blake3(&self) -> blake3::Hash {
        blake3::Hash::from_bytes(self.canonical_blake3)
    }

    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(CATALOG_MAGIC.len() + 4 + self.definitions.len() * 14);
        bytes.extend_from_slice(CATALOG_MAGIC);
        bytes.extend_from_slice(&EXACT_SEMANTIC_SUPPLY_SCHEMA_VERSION.to_le_bytes());
        bytes.extend_from_slice(&(self.definitions.len() as u16).to_le_bytes());
        for definition in &self.definitions {
            bytes.extend_from_slice(&definition.id.code().to_le_bytes());
            bytes.extend_from_slice(&definition.archetype.canonical_bytes());
            bytes.extend_from_slice(&definition.standard_tile_count.to_le_bytes());
        }
        bytes
    }

    pub fn definition(&self, id: SemanticArchetypeId) -> Option<&SemanticArchetypeDefinition> {
        self.definitions.get(id.index())
    }

    pub fn id_for_archetype(
        &self,
        archetype: CanonicalTileArchetype,
    ) -> Option<SemanticArchetypeId> {
        let key = archetype.canonical_bytes();
        self.definitions
            .binary_search_by_key(&key, |definition| definition.archetype.canonical_bytes())
            .ok()
            .map(|index| self.definitions[index].id)
    }

    pub fn reference_for_tile(
        &self,
        tile: Tile,
    ) -> Result<SemanticTileReference, SemanticSupplyError> {
        let archetype = CanonicalTileArchetype::from_tile(tile);
        let archetype_id = self
            .id_for_archetype(archetype)
            .ok_or(SemanticSupplyError::UnknownTileArchetype)?;
        let canonical_edges = terrain_codes(archetype.directed_edges);
        let tile_to_canonical_rotation = Rotation::ALL
            .into_iter()
            .find(|rotation| {
                terrain_codes(std::array::from_fn(|edge| {
                    tile.terrain_on_edge(*rotation, edge)
                })) == canonical_edges
            })
            .ok_or(SemanticSupplyError::UnknownTileArchetype)?;
        Ok(SemanticTileReference {
            archetype_id,
            tile_to_canonical_rotation,
        })
    }

    fn standard_counts(&self) -> Vec<u16> {
        self.definitions
            .iter()
            .map(|definition| definition.standard_tile_count)
            .collect()
    }
}

pub fn standard_semantic_archetype_catalog() -> &'static SemanticArchetypeCatalog {
    static CATALOG: OnceLock<SemanticArchetypeCatalog> = OnceLock::new();
    CATALOG.get_or_init(|| build_catalog(STANDARD_TILES))
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExactSemanticSupply {
    wildlife_bag: [u16; 5],
    archetype_counts: Vec<u16>,
    unseen_tile_count: u16,
    drawable_tile_count: u16,
    catalog_blake3: [u8; 32],
}

impl ExactSemanticSupply {
    pub fn from_game(game: &GameState) -> Result<Self, SemanticSupplyError> {
        Self::from_public_state(&game.public_state())
    }

    pub fn from_public_state(state: &PublicGameState) -> Result<Self, SemanticSupplyError> {
        if state.config().mode != GameMode::Standard {
            return Err(SemanticSupplyError::UnsupportedPublicMode(
                state.config().mode,
            ));
        }

        let catalog = standard_semantic_archetype_catalog();
        let mut archetype_counts = catalog.standard_counts();
        let mut visible_tile_ids = [false; STANDARD_TILES.len()];
        for tile in state
            .boards()
            .iter()
            .flat_map(|board| board.placed_tiles().map(|(_, placed)| placed.tile))
            .chain(state.market().tiles.iter().flatten().copied())
        {
            remove_publicly_visible_standard_tile(
                tile,
                &mut visible_tile_ids,
                &mut archetype_counts,
            )?;
        }
        let visible_standard_tiles = visible_tile_ids
            .into_iter()
            .filter(|visible| *visible)
            .count() as u16;
        // Standard multiplayer starts with exactly total_turns + 3 drawable
        // tiles: four enter the opening market and one refill follows every
        // turn except the last.
        let drawable_tile_count = state
            .total_turns()
            .checked_add(3)
            .and_then(|initial| initial.checked_sub(visible_standard_tiles))
            .ok_or(SemanticSupplyError::DrawableTileConservation)?;

        let mut wildlife_bag = [20u16; 5];
        for wildlife in state
            .boards()
            .iter()
            .flat_map(|board| {
                board
                    .placed_tiles()
                    .filter_map(|(_, placed)| placed.wildlife)
            })
            .chain(state.market().wildlife.iter().flatten().copied())
        {
            wildlife_bag[wildlife as usize] = wildlife_bag[wildlife as usize]
                .checked_sub(1)
                .ok_or(SemanticSupplyError::WildlifeConservation(wildlife))?;
        }

        Self::from_exact_counts(wildlife_bag, archetype_counts, drawable_tile_count)
    }

    pub fn wildlife_bag_counts(&self) -> [u16; 5] {
        self.wildlife_bag
    }

    pub fn archetype_counts(&self) -> &[u16] {
        &self.archetype_counts
    }

    pub fn count(&self, id: SemanticArchetypeId) -> Option<u16> {
        self.archetype_counts.get(id.index()).copied()
    }

    pub const fn unseen_tile_count(&self) -> u16 {
        self.unseen_tile_count
    }

    pub const fn drawable_tile_count(&self) -> u16 {
        self.drawable_tile_count
    }

    pub const fn excluded_tile_count(&self) -> u16 {
        self.unseen_tile_count - self.drawable_tile_count
    }

    pub fn catalog_blake3(&self) -> blake3::Hash {
        blake3::Hash::from_bytes(self.catalog_blake3)
    }

    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes =
            Vec::with_capacity(SUPPLY_MAGIC.len() + 50 + self.archetype_counts.len() * 2);
        bytes.extend_from_slice(SUPPLY_MAGIC);
        bytes.extend_from_slice(&EXACT_SEMANTIC_SUPPLY_SCHEMA_VERSION.to_le_bytes());
        bytes.extend_from_slice(&self.catalog_blake3);
        for count in self.wildlife_bag {
            bytes.extend_from_slice(&count.to_le_bytes());
        }
        bytes.extend_from_slice(&self.unseen_tile_count.to_le_bytes());
        bytes.extend_from_slice(&self.drawable_tile_count.to_le_bytes());
        bytes.extend_from_slice(&(self.archetype_counts.len() as u16).to_le_bytes());
        for count in &self.archetype_counts {
            bytes.extend_from_slice(&count.to_le_bytes());
        }
        bytes
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, SemanticSupplyError> {
        let mut cursor = ByteCursor::new(bytes);
        if cursor.take_array::<8>()? != *SUPPLY_MAGIC {
            return Err(SemanticSupplyError::InvalidSerialization(
                "invalid semantic supply magic",
            ));
        }
        if cursor.take_u16()? != EXACT_SEMANTIC_SUPPLY_SCHEMA_VERSION {
            return Err(SemanticSupplyError::InvalidSerialization(
                "unsupported semantic supply schema",
            ));
        }
        let catalog_blake3 = cursor.take_array::<32>()?;
        let mut wildlife_bag = [0u16; 5];
        for count in &mut wildlife_bag {
            *count = cursor.take_u16()?;
        }
        let declared_total = cursor.take_u16()?;
        let drawable_tile_count = cursor.take_u16()?;
        let count_len = usize::from(cursor.take_u16()?);
        let archetype_counts = (0..count_len)
            .map(|_| cursor.take_u16())
            .collect::<Result<Vec<_>, _>>()?;
        cursor.finish()?;

        let supply = Self::from_exact_counts(wildlife_bag, archetype_counts, drawable_tile_count)?;
        if supply.catalog_blake3 != catalog_blake3 || supply.unseen_tile_count != declared_total {
            return Err(SemanticSupplyError::InvalidSerialization(
                "semantic supply identity mismatch",
            ));
        }
        Ok(supply)
    }

    pub fn canonical_hash(&self) -> blake3::Hash {
        blake3::hash(&self.canonical_bytes())
    }

    pub fn to_legacy_public_supply(&self) -> PublicSupply {
        let mut supply = PublicSupply {
            wildlife_bag: self.wildlife_bag.map(|count| count as u8),
            unseen_tile_terrain_capacity: [0; 5],
            unseen_tile_wildlife_capacity: [0; 5],
            unseen_keystones_by_terrain: [0; 5],
            unseen_dual_terrain_pairs: [0; 10],
        };
        let catalog = standard_semantic_archetype_catalog();
        for (definition, count) in catalog.definitions().iter().zip(&self.archetype_counts) {
            let count = *count as u8;
            let archetype = definition.archetype;
            for terrain in Terrain::ALL {
                if archetype.contains_terrain(terrain) {
                    supply.unseen_tile_terrain_capacity[terrain as usize] += count;
                }
            }
            for wildlife in Wildlife::ALL {
                if archetype.wildlife.contains(wildlife) {
                    supply.unseen_tile_wildlife_capacity[wildlife as usize] += count;
                }
            }
            if archetype.keystone {
                supply.unseen_keystones_by_terrain[archetype.primary_terrain as usize] += count;
            } else if let Some(secondary) = archetype.secondary_terrain {
                supply.unseen_dual_terrain_pairs
                    [terrain_pair_index(archetype.primary_terrain, secondary)] += count;
            }
        }
        supply
    }

    pub fn refill_distribution(
        &self,
        slot_count: u8,
    ) -> Result<ExactRefillDistribution, SemanticSupplyError> {
        if u16::from(slot_count) > self.drawable_tile_count {
            return Err(SemanticSupplyError::RefillExceedsSupply {
                requested: slot_count,
                available: self.drawable_tile_count,
            });
        }
        ExactRefillDistribution::new(
            self.catalog_blake3,
            self.archetype_counts.clone(),
            slot_count,
        )
    }

    pub fn market_links(
        &self,
        market: &Market,
    ) -> Result<[Option<SemanticMarketLink>; 4], SemanticSupplyError> {
        let catalog = standard_semantic_archetype_catalog();
        let mut links = [None; 4];
        for slot in MarketSlot::ALL {
            if let Some(tile) = market.tiles[slot.index()] {
                links[slot.index()] = Some(SemanticMarketLink {
                    slot,
                    tile: catalog.reference_for_tile(tile)?,
                });
            }
        }
        Ok(links)
    }

    fn from_exact_counts(
        wildlife_bag: [u16; 5],
        archetype_counts: Vec<u16>,
        drawable_tile_count: u16,
    ) -> Result<Self, SemanticSupplyError> {
        let catalog = standard_semantic_archetype_catalog();
        if archetype_counts.len() != catalog.len() {
            return Err(SemanticSupplyError::ArchetypeCountLength {
                expected: catalog.len(),
                actual: archetype_counts.len(),
            });
        }
        for (definition, count) in catalog.definitions().iter().zip(&archetype_counts) {
            if *count > definition.standard_tile_count {
                return Err(SemanticSupplyError::ArchetypeCountExceedsCatalog {
                    id: definition.id,
                    count: *count,
                    maximum: definition.standard_tile_count,
                });
            }
        }
        for (index, count) in wildlife_bag.into_iter().enumerate() {
            if count > 20 {
                return Err(SemanticSupplyError::WildlifeCountExceedsCatalog {
                    wildlife: Wildlife::ALL[index],
                    count,
                });
            }
        }
        let unseen_tile_count = archetype_counts.iter().sum();
        if drawable_tile_count > unseen_tile_count {
            return Err(SemanticSupplyError::DrawableTileCountExceedsUnseen {
                drawable: drawable_tile_count,
                unseen: unseen_tile_count,
            });
        }
        Ok(Self {
            wildlife_bag,
            archetype_counts,
            unseen_tile_count,
            drawable_tile_count,
            catalog_blake3: *catalog.canonical_blake3().as_bytes(),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArchetypeProbability {
    pub archetype_id: SemanticArchetypeId,
    pub probability: ExactProbability,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExactProbability {
    pub numerator: u64,
    pub denominator: u64,
}

impl ExactProbability {
    pub const ZERO: Self = Self {
        numerator: 0,
        denominator: 1,
    };
    pub const ONE: Self = Self {
        numerator: 1,
        denominator: 1,
    };

    pub fn new(numerator: u64, denominator: u64) -> Result<Self, SemanticSupplyError> {
        if denominator == 0 {
            return Err(SemanticSupplyError::ZeroProbabilityDenominator);
        }
        if numerator == 0 {
            return Ok(Self::ZERO);
        }
        let divisor = gcd(numerator, denominator);
        Ok(Self {
            numerator: numerator / divisor,
            denominator: denominator / divisor,
        })
    }

    pub fn as_f64(self) -> f64 {
        self.numerator as f64 / self.denominator as f64
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OrderedRefillOutcome {
    pub archetypes: Vec<SemanticArchetypeId>,
    pub ordered_weight: u64,
    pub probability: ExactProbability,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExactRefillDistribution {
    catalog_blake3: [u8; 32],
    archetype_counts: Vec<u16>,
    total_unseen: u16,
    slot_count: u8,
}

impl ExactRefillDistribution {
    fn new(
        catalog_blake3: [u8; 32],
        archetype_counts: Vec<u16>,
        slot_count: u8,
    ) -> Result<Self, SemanticSupplyError> {
        if slot_count == 0 || slot_count > MAX_EXACT_REFILL_SLOTS {
            return Err(SemanticSupplyError::InvalidRefillSlotCount(slot_count));
        }
        if archetype_counts.len() != standard_semantic_archetype_catalog().len() {
            return Err(SemanticSupplyError::ArchetypeCountLength {
                expected: standard_semantic_archetype_catalog().len(),
                actual: archetype_counts.len(),
            });
        }
        let total_unseen: u16 = archetype_counts.iter().sum();
        if u16::from(slot_count) > total_unseen {
            return Err(SemanticSupplyError::RefillExceedsSupply {
                requested: slot_count,
                available: total_unseen,
            });
        }
        Ok(Self {
            catalog_blake3,
            archetype_counts,
            total_unseen,
            slot_count,
        })
    }

    pub const fn slot_count(&self) -> u8 {
        self.slot_count
    }

    pub const fn total_unseen(&self) -> u16 {
        self.total_unseen
    }

    pub fn catalog_blake3(&self) -> blake3::Hash {
        blake3::Hash::from_bytes(self.catalog_blake3)
    }

    pub fn ordered_denominator(&self) -> u64 {
        falling_factorial(u64::from(self.total_unseen), self.slot_count)
    }

    pub fn one_slot_probabilities(&self) -> Vec<ArchetypeProbability> {
        self.archetype_counts
            .iter()
            .enumerate()
            .map(|(index, count)| ArchetypeProbability {
                archetype_id: SemanticArchetypeId(index as u16),
                probability: ExactProbability::new(u64::from(*count), u64::from(self.total_unseen))
                    .expect("a refill distribution has a positive total"),
            })
            .collect()
    }

    pub fn ordered_sequence_weight(
        &self,
        sequence: &[SemanticArchetypeId],
    ) -> Result<u64, SemanticSupplyError> {
        if sequence.len() != usize::from(self.slot_count) {
            return Err(SemanticSupplyError::RefillSequenceLength {
                expected: usize::from(self.slot_count),
                actual: sequence.len(),
            });
        }
        let mut remaining = self.archetype_counts.clone();
        let mut weight = 1u64;
        for id in sequence {
            let available = remaining
                .get_mut(id.index())
                .ok_or(SemanticSupplyError::UnknownArchetypeId(*id))?;
            if *available == 0 {
                return Ok(0);
            }
            weight = weight
                .checked_mul(u64::from(*available))
                .ok_or(SemanticSupplyError::ProbabilityOverflow)?;
            *available -= 1;
        }
        Ok(weight)
    }

    pub fn probability_of_ordered(
        &self,
        sequence: &[SemanticArchetypeId],
    ) -> Result<ExactProbability, SemanticSupplyError> {
        ExactProbability::new(
            self.ordered_sequence_weight(sequence)?,
            self.ordered_denominator(),
        )
    }

    pub fn probability_of_unordered(
        &self,
        requested_counts: &[(SemanticArchetypeId, u8)],
    ) -> Result<ExactProbability, SemanticSupplyError> {
        let mut requested = vec![0u8; self.archetype_counts.len()];
        let mut requested_total = 0u8;
        for (id, count) in requested_counts {
            let slot = requested
                .get_mut(id.index())
                .ok_or(SemanticSupplyError::UnknownArchetypeId(*id))?;
            if *slot != 0 {
                return Err(SemanticSupplyError::DuplicateArchetypeRequest(*id));
            }
            *slot = *count;
            requested_total = requested_total
                .checked_add(*count)
                .ok_or(SemanticSupplyError::ProbabilityOverflow)?;
        }
        if requested_total != self.slot_count {
            return Err(SemanticSupplyError::RefillSequenceLength {
                expected: usize::from(self.slot_count),
                actual: usize::from(requested_total),
            });
        }

        let mut numerator = 1u64;
        for (available, count) in self.archetype_counts.iter().zip(requested) {
            if u16::from(count) > *available {
                return Ok(ExactProbability::ZERO);
            }
            numerator = numerator
                .checked_mul(binomial(u64::from(*available), u64::from(count)))
                .ok_or(SemanticSupplyError::ProbabilityOverflow)?;
        }
        ExactProbability::new(
            numerator,
            binomial(u64::from(self.total_unseen), u64::from(self.slot_count)),
        )
    }

    pub fn conditional_after(
        &self,
        prefix: &[SemanticArchetypeId],
    ) -> Result<Self, SemanticSupplyError> {
        if prefix.len() >= usize::from(self.slot_count) {
            return Err(SemanticSupplyError::ConditionalPrefixLength {
                slots: self.slot_count,
                prefix: prefix.len(),
            });
        }
        let mut counts = self.archetype_counts.clone();
        for id in prefix {
            let count = counts
                .get_mut(id.index())
                .ok_or(SemanticSupplyError::UnknownArchetypeId(*id))?;
            if *count == 0 {
                return Err(SemanticSupplyError::ImpossibleRefillPrefix(*id));
            }
            *count -= 1;
        }
        Self::new(
            self.catalog_blake3,
            counts,
            self.slot_count - prefix.len() as u8,
        )
    }

    pub fn enumerate_ordered_outcomes(
        &self,
        maximum_outcomes: usize,
    ) -> Result<Vec<OrderedRefillOutcome>, SemanticSupplyError> {
        if maximum_outcomes == 0 {
            return Err(SemanticSupplyError::OutcomeLimitExceeded {
                maximum: maximum_outcomes,
            });
        }
        let mut outcomes = Vec::new();
        let mut counts = self.archetype_counts.clone();
        let mut prefix = Vec::with_capacity(usize::from(self.slot_count));
        enumerate_outcomes(
            &mut counts,
            self.slot_count,
            1,
            &mut prefix,
            self.ordered_denominator(),
            maximum_outcomes,
            &mut outcomes,
        )?;
        Ok(outcomes)
    }

    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes =
            Vec::with_capacity(REFILL_MAGIC.len() + 39 + self.archetype_counts.len() * 2);
        bytes.extend_from_slice(REFILL_MAGIC);
        bytes.extend_from_slice(&EXACT_SEMANTIC_SUPPLY_SCHEMA_VERSION.to_le_bytes());
        bytes.push(self.slot_count);
        bytes.extend_from_slice(&self.catalog_blake3);
        bytes.extend_from_slice(&(self.archetype_counts.len() as u16).to_le_bytes());
        for count in &self.archetype_counts {
            bytes.extend_from_slice(&count.to_le_bytes());
        }
        bytes
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, SemanticSupplyError> {
        let mut cursor = ByteCursor::new(bytes);
        if cursor.take_array::<8>()? != *REFILL_MAGIC {
            return Err(SemanticSupplyError::InvalidSerialization(
                "invalid refill distribution magic",
            ));
        }
        if cursor.take_u16()? != EXACT_SEMANTIC_SUPPLY_SCHEMA_VERSION {
            return Err(SemanticSupplyError::InvalidSerialization(
                "unsupported refill distribution schema",
            ));
        }
        let slot_count = cursor.take_u8()?;
        let catalog_blake3 = cursor.take_array::<32>()?;
        let count_len = usize::from(cursor.take_u16()?);
        let archetype_counts = (0..count_len)
            .map(|_| cursor.take_u16())
            .collect::<Result<Vec<_>, _>>()?;
        cursor.finish()?;
        if catalog_blake3
            != *standard_semantic_archetype_catalog()
                .canonical_blake3()
                .as_bytes()
        {
            return Err(SemanticSupplyError::InvalidSerialization(
                "refill catalog identity mismatch",
            ));
        }
        Self::new(catalog_blake3, archetype_counts, slot_count)
    }

    pub fn canonical_hash(&self) -> blake3::Hash {
        blake3::hash(&self.canonical_bytes())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum SemanticSupplyError {
    #[error("exact public semantic supply does not support {0:?} without public discard history")]
    UnsupportedPublicMode(GameMode),
    #[error("public state repeats standard habitat tile ID {0}")]
    DuplicateVisibleTileId(u8),
    #[error("public tile ID {id} does not match the official standard catalog")]
    PublicTileIdentityMismatch { id: u8 },
    #[error("publicly visible tile exhausted archetype {0:?}")]
    VisibleTileCountUnderflow(SemanticArchetypeId),
    #[error("wildlife conservation failed for {0:?}")]
    WildlifeConservation(Wildlife),
    #[error("public drawable tile conservation failed")]
    DrawableTileConservation,
    #[error("tile semantics are not present in the official standard catalog")]
    UnknownTileArchetype,
    #[error("semantic tile reference does not match the supplied tile")]
    TileReferenceMismatch,
    #[error("unknown semantic archetype ID {0:?}")]
    UnknownArchetypeId(SemanticArchetypeId),
    #[error("archetype count vector has length {actual}; expected {expected}")]
    ArchetypeCountLength { expected: usize, actual: usize },
    #[error("archetype {id:?} count {count} exceeds catalog multiplicity {maximum}")]
    ArchetypeCountExceedsCatalog {
        id: SemanticArchetypeId,
        count: u16,
        maximum: u16,
    },
    #[error("{wildlife:?} bag count {count} exceeds the official supply of 20")]
    WildlifeCountExceedsCatalog { wildlife: Wildlife, count: u16 },
    #[error("drawable tile count {drawable} exceeds unseen tile count {unseen}")]
    DrawableTileCountExceedsUnseen { drawable: u16, unseen: u16 },
    #[error("refill slot count {0} must be in 1..={MAX_EXACT_REFILL_SLOTS}")]
    InvalidRefillSlotCount(u8),
    #[error("cannot refill {requested} slots from {available} drawable tiles")]
    RefillExceedsSupply { requested: u8, available: u16 },
    #[error("refill sequence has length {actual}; expected {expected}")]
    RefillSequenceLength { expected: usize, actual: usize },
    #[error("conditional prefix length {prefix} must be less than the {slots}-slot horizon")]
    ConditionalPrefixLength { slots: u8, prefix: usize },
    #[error("conditional prefix requests unavailable archetype {0:?}")]
    ImpossibleRefillPrefix(SemanticArchetypeId),
    #[error("unordered refill request repeats archetype {0:?}")]
    DuplicateArchetypeRequest(SemanticArchetypeId),
    #[error("probability denominator must be positive")]
    ZeroProbabilityDenominator,
    #[error("exact probability arithmetic overflowed")]
    ProbabilityOverflow,
    #[error("ordered refill support exceeds configured limit {maximum}")]
    OutcomeLimitExceeded { maximum: usize },
    #[error("invalid exact semantic supply serialization: {0}")]
    InvalidSerialization(&'static str),
    #[error(transparent)]
    D6(#[from] cascadia_game::D6Error),
}

fn build_catalog(tiles: impl IntoIterator<Item = Tile>) -> SemanticArchetypeCatalog {
    let mut archetypes: Vec<_> = tiles
        .into_iter()
        .map(CanonicalTileArchetype::from_tile)
        .collect();
    archetypes.sort_unstable_by_key(|archetype| archetype.canonical_bytes());

    let mut definitions: Vec<SemanticArchetypeDefinition> = Vec::new();
    for archetype in archetypes {
        if let Some(last) = definitions.last_mut()
            && last.archetype == archetype
        {
            last.standard_tile_count += 1;
            continue;
        }
        definitions.push(SemanticArchetypeDefinition {
            id: SemanticArchetypeId(definitions.len() as u16),
            archetype,
            standard_tile_count: 1,
        });
    }

    let mut catalog = SemanticArchetypeCatalog {
        definitions,
        canonical_blake3: [0; 32],
    };
    catalog.canonical_blake3 = *blake3::hash(&catalog.canonical_bytes()).as_bytes();
    catalog
}

fn canonical_terrains(tile: Tile) -> (Terrain, Option<Terrain>) {
    match tile.terrain_b {
        Some(secondary) if secondary as u8 != tile.terrain_a as u8 => {
            if (tile.terrain_a as u8) < (secondary as u8) {
                (tile.terrain_a, Some(secondary))
            } else {
                (secondary, Some(tile.terrain_a))
            }
        }
        _ => (tile.terrain_a, None),
    }
}

fn canonical_edge_ring(tile: Tile) -> [Terrain; 6] {
    let mut best = terrain_codes(std::array::from_fn(|edge| {
        tile.terrain_on_edge(Rotation::ZERO, edge)
    }));
    for rotation in Rotation::ALL.into_iter().skip(1) {
        let candidate = terrain_codes(std::array::from_fn(|edge| {
            tile.terrain_on_edge(rotation, edge)
        }));
        if candidate < best {
            best = candidate;
        }
    }
    best.map(|code| decode_terrain(code).expect("tile terrains are canonical"))
}

fn terrain_codes(terrains: [Terrain; 6]) -> [u8; 6] {
    terrains.map(|terrain| terrain as u8)
}

fn decode_terrain(code: u8) -> Option<Terrain> {
    Terrain::ALL
        .into_iter()
        .find(|terrain| *terrain as u8 == code)
}

fn remove_publicly_visible_standard_tile(
    tile: Tile,
    visible_tile_ids: &mut [bool; STANDARD_TILES.len()],
    archetype_counts: &mut [u16],
) -> Result<(), SemanticSupplyError> {
    let id = usize::from(tile.id.0);
    if id >= STANDARD_TILES.len() {
        return Ok(());
    }
    if STANDARD_TILES[id] != tile {
        return Err(SemanticSupplyError::PublicTileIdentityMismatch { id: tile.id.0 });
    }
    if std::mem::replace(&mut visible_tile_ids[id], true) {
        return Err(SemanticSupplyError::DuplicateVisibleTileId(tile.id.0));
    }
    let reference = standard_semantic_archetype_catalog().reference_for_tile(tile)?;
    archetype_counts[reference.archetype_id.index()] = archetype_counts
        [reference.archetype_id.index()]
    .checked_sub(1)
    .ok_or(SemanticSupplyError::VisibleTileCountUnderflow(
        reference.archetype_id,
    ))?;
    Ok(())
}

fn frontier_compatibility(
    tile: Tile,
    requirements: FrontierTerrainRequirements,
) -> FrontierCompatibility {
    let present_edges = requirements.present_edges();
    let matching_edges_by_rotation = Rotation::ALL.map(|rotation| {
        requirements
            .neighbor_facing_terrains
            .into_iter()
            .enumerate()
            .filter(|(edge, required)| {
                required.is_some_and(|terrain| tile.terrain_on_edge(rotation, *edge) == terrain)
            })
            .count() as u8
    });
    let mut all_present_match_rotation_mask = 0u8;
    for rotation in Rotation::ALL {
        if matching_edges_by_rotation[usize::from(rotation.get())] == present_edges {
            all_present_match_rotation_mask |= 1 << rotation.get();
        }
    }
    FrontierCompatibility {
        matching_edges_by_rotation,
        all_present_match_rotation_mask,
        best_matching_edges: matching_edges_by_rotation.into_iter().max().unwrap_or(0),
    }
}

fn terrain_pair_index(left: Terrain, right: Terrain) -> usize {
    let (low, high) = if (left as u8) < (right as u8) {
        (left as usize, right as usize)
    } else {
        (right as usize, left as usize)
    };
    let mut index = 0;
    for first in 0..5 {
        for second in first + 1..5 {
            if first == low && second == high {
                return index;
            }
            index += 1;
        }
    }
    unreachable!("canonical dual-terrain archetype has distinct terrains")
}

fn enumerate_outcomes(
    counts: &mut [u16],
    slots_remaining: u8,
    ordered_weight: u64,
    prefix: &mut Vec<SemanticArchetypeId>,
    denominator: u64,
    maximum_outcomes: usize,
    outcomes: &mut Vec<OrderedRefillOutcome>,
) -> Result<(), SemanticSupplyError> {
    if slots_remaining == 0 {
        if outcomes.len() == maximum_outcomes {
            return Err(SemanticSupplyError::OutcomeLimitExceeded {
                maximum: maximum_outcomes,
            });
        }
        outcomes.push(OrderedRefillOutcome {
            archetypes: prefix.clone(),
            ordered_weight,
            probability: ExactProbability::new(ordered_weight, denominator)?,
        });
        return Ok(());
    }

    for index in 0..counts.len() {
        let available = counts[index];
        if available == 0 {
            continue;
        }
        counts[index] -= 1;
        prefix.push(SemanticArchetypeId(index as u16));
        enumerate_outcomes(
            counts,
            slots_remaining - 1,
            ordered_weight
                .checked_mul(u64::from(available))
                .ok_or(SemanticSupplyError::ProbabilityOverflow)?,
            prefix,
            denominator,
            maximum_outcomes,
            outcomes,
        )?;
        prefix.pop();
        counts[index] += 1;
    }
    Ok(())
}

const fn falling_factorial(total: u64, slots: u8) -> u64 {
    let mut result = 1u64;
    let mut offset = 0u8;
    while offset < slots {
        result *= total - offset as u64;
        offset += 1;
    }
    result
}

const fn binomial(n: u64, k: u64) -> u64 {
    if k > n {
        return 0;
    }
    let k = if k < n - k { k } else { n - k };
    let mut result = 1u64;
    let mut index = 0u64;
    while index < k {
        result = result * (n - index) / (index + 1);
        index += 1;
    }
    result
}

const fn gcd(mut left: u64, mut right: u64) -> u64 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

struct ByteCursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> ByteCursor<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn take_u8(&mut self) -> Result<u8, SemanticSupplyError> {
        Ok(self.take_array::<1>()?[0])
    }

    fn take_u16(&mut self) -> Result<u16, SemanticSupplyError> {
        Ok(u16::from_le_bytes(self.take_array()?))
    }

    fn take_array<const N: usize>(&mut self) -> Result<[u8; N], SemanticSupplyError> {
        let end = self
            .offset
            .checked_add(N)
            .ok_or(SemanticSupplyError::InvalidSerialization(
                "serialized length overflow",
            ))?;
        let value = self
            .bytes
            .get(self.offset..end)
            .ok_or(SemanticSupplyError::InvalidSerialization(
                "truncated semantic supply",
            ))?
            .try_into()
            .expect("slice length was checked");
        self.offset = end;
        Ok(value)
    }

    fn finish(self) -> Result<(), SemanticSupplyError> {
        if self.offset == self.bytes.len() {
            Ok(())
        } else {
            Err(SemanticSupplyError::InvalidSerialization(
                "trailing semantic supply bytes",
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use cascadia_game::{
        D6Transform, DraftChoice, GameConfig, GameSeed, HexDirection, MarketPrelude, ScoringCards,
    };

    use super::*;

    fn game(seed: u64) -> GameState {
        GameState::new(
            GameConfig::research_aaaaa(4).unwrap(),
            GameSeed::from_u64(seed),
        )
        .unwrap()
    }

    fn advance_deterministically(game: &mut GameState, turns: usize) {
        for turn in 0..turns {
            let actions = game.legal_turn_actions(&MarketPrelude::default()).unwrap();
            let action = actions[(turn * 7_919 + 17) % actions.len()].clone();
            game.apply(&action).unwrap();
        }
    }

    #[test]
    fn standard_catalog_is_orientation_normalized_complete_and_deterministic() {
        let catalog = standard_semantic_archetype_catalog();
        assert_eq!(catalog.len(), 75);
        assert_eq!(
            catalog.canonical_blake3().to_hex().as_str(),
            "362a1f090066f537fc29398fdc464f667b7e106889feff8a77607e35dd015c19"
        );
        assert_eq!(
            catalog
                .definitions()
                .iter()
                .map(|definition| usize::from(definition.standard_tile_count))
                .sum::<usize>(),
            STANDARD_TILES.len()
        );
        assert!(
            catalog
                .definitions()
                .windows(2)
                .all(|pair| pair[0].archetype.canonical_bytes()
                    < pair[1].archetype.canonical_bytes())
        );

        let reversed = build_catalog(STANDARD_TILES.into_iter().rev());
        assert_eq!(reversed.definitions, catalog.definitions);
        assert_eq!(reversed.canonical_blake3(), catalog.canonical_blake3());

        for tile in STANDARD_TILES {
            let swapped = match tile.terrain_b {
                Some(secondary) => {
                    Tile::dual(tile.id.0, secondary, tile.terrain_a, tile.wildlife.bits())
                }
                None => tile,
            };
            assert_eq!(
                CanonicalTileArchetype::from_tile(tile),
                CanonicalTileArchetype::from_tile(swapped)
            );
            let reference = catalog.reference_for_tile(tile).unwrap();
            for game_rotation in Rotation::ALL {
                let canonical_rotation = reference.canonical_rotation_for_game(game_rotation);
                for edge in 0..6 {
                    assert_eq!(
                        catalog
                            .definition(reference.archetype_id)
                            .unwrap()
                            .archetype
                            .terrain_on_edge(canonical_rotation, edge),
                        tile.terrain_on_edge(game_rotation, edge)
                    );
                }
                assert_eq!(
                    reference.game_rotation_for_canonical(canonical_rotation),
                    game_rotation
                );
            }
        }
    }

    #[test]
    fn exact_supply_uses_only_public_observations_and_matches_legacy_marginals() {
        for seed in 0..8 {
            let mut state = game(seed);
            for turn in 0..12 {
                let exact = ExactSemanticSupply::from_game(&state).unwrap();
                assert_eq!(exact.to_legacy_public_supply(), state.public_supply());
                assert_eq!(exact.unseen_tile_count(), 81u16.saturating_sub(turn as u16));
                assert_eq!(
                    exact.drawable_tile_count(),
                    79u16.saturating_sub(turn as u16)
                );
                assert_eq!(exact.excluded_tile_count(), 2);
                let action = state
                    .legal_turn_actions(&MarketPrelude::default())
                    .unwrap()
                    .into_iter()
                    .next()
                    .unwrap();
                state.apply(&action).unwrap();
            }
        }

        let mut final_turn = game(8_888);
        advance_deterministically(&mut final_turn, 79);
        let supply = ExactSemanticSupply::from_game(&final_turn).unwrap();
        assert_eq!(supply.unseen_tile_count(), 2);
        assert_eq!(supply.drawable_tile_count(), 0);
        assert_eq!(
            supply.refill_distribution(1),
            Err(SemanticSupplyError::RefillExceedsSupply {
                requested: 1,
                available: 0,
            })
        );
    }

    #[test]
    fn drawable_and_excluded_counts_match_each_standard_player_count() {
        for (player_count, drawable, excluded) in [(2, 39, 42), (3, 59, 22), (4, 79, 2)] {
            let state = GameState::new(
                GameConfig::research_aaaaa(player_count).unwrap(),
                GameSeed::from_u64(90_000 + u64::from(player_count)),
            )
            .unwrap();
            let supply = ExactSemanticSupply::from_game(&state).unwrap();
            assert_eq!(supply.unseen_tile_count(), 81);
            assert_eq!(supply.drawable_tile_count(), drawable);
            assert_eq!(supply.excluded_tile_count(), excluded);
        }
    }

    #[test]
    fn public_board_permutation_and_afterstate_staging_preserve_exact_supply() {
        let mut state = game(37);
        advance_deterministically(&mut state, 9);
        let expected = ExactSemanticSupply::from_game(&state).unwrap();

        let mut public_json = serde_json::to_value(state.public_state()).unwrap();
        public_json["boards"].as_array_mut().unwrap().reverse();
        let permuted: PublicGameState = serde_json::from_value(public_json).unwrap();
        assert_eq!(
            ExactSemanticSupply::from_public_state(&permuted).unwrap(),
            expected
        );

        let action = state
            .legal_turn_actions(&MarketPrelude::default())
            .unwrap()
            .into_iter()
            .next()
            .unwrap();
        let afterstate = state.preview_public_afterstate(&action).unwrap();
        let staged = ExactSemanticSupply::from_public_state(&afterstate).unwrap();
        assert_eq!(staged.archetype_counts(), expected.archetype_counts());
        assert_eq!(staged.drawable_tile_count(), expected.drawable_tile_count());
        let mut expected_wildlife = expected.wildlife_bag_counts();
        if action.wildlife.is_none() {
            let wildlife_slot = match action.draft {
                DraftChoice::Paired { slot } => slot,
                DraftChoice::Independent { wildlife_slot, .. } => wildlife_slot,
            };
            let wildlife = state.market().wildlife[wildlife_slot.index()].unwrap();
            expected_wildlife[wildlife as usize] += 1;
        }
        assert_eq!(staged.wildlife_bag_counts(), expected_wildlife);
        assert_eq!(
            staged
                .market_links(afterstate.market())
                .unwrap()
                .iter()
                .filter(|link| link.is_some())
                .count(),
            3
        );
    }

    #[test]
    fn hidden_order_and_d6_do_not_change_supply_or_refill_law() {
        let mut state = game(91);
        advance_deterministically(&mut state, 11);
        let expected = ExactSemanticSupply::from_game(&state).unwrap();
        let expected_refill = expected.refill_distribution(4).unwrap();

        for seed in 0..16 {
            let mut redetermined = state.clone();
            redetermined.redeterminize_hidden(GameSeed::from_u64(50_000 + seed));
            let actual = ExactSemanticSupply::from_game(&redetermined).unwrap();
            assert_eq!(actual, expected);
            assert_eq!(
                actual.refill_distribution(4).unwrap().canonical_hash(),
                expected_refill.canonical_hash()
            );
        }

        for transform in D6Transform::ALL {
            let transformed = state.transformed(transform).unwrap();
            assert_eq!(
                ExactSemanticSupply::from_game(&transformed).unwrap(),
                expected
            );
        }
    }

    #[test]
    fn supply_and_refill_serialization_round_trip_and_reject_drift() {
        let supply = ExactSemanticSupply::from_game(&game(12)).unwrap();
        let bytes = supply.canonical_bytes();
        assert_eq!(
            ExactSemanticSupply::from_canonical_bytes(&bytes).unwrap(),
            supply
        );
        let mut drifted = bytes;
        *drifted.last_mut().unwrap() = u8::MAX;
        assert!(ExactSemanticSupply::from_canonical_bytes(&drifted).is_err());
        let mut impossible_drawable = supply.canonical_bytes();
        let drawable_offset = SUPPLY_MAGIC.len() + 2 + 32 + 5 * 2 + 2;
        impossible_drawable[drawable_offset..drawable_offset + 2]
            .copy_from_slice(&(supply.unseen_tile_count() + 1).to_le_bytes());
        assert!(matches!(
            ExactSemanticSupply::from_canonical_bytes(&impossible_drawable),
            Err(SemanticSupplyError::DrawableTileCountExceedsUnseen { .. })
        ));

        for slots in 1..=4 {
            let refill = supply.refill_distribution(slots).unwrap();
            assert_eq!(
                ExactRefillDistribution::from_canonical_bytes(&refill.canonical_bytes()).unwrap(),
                refill
            );
        }
    }

    #[test]
    fn drawable_horizon_limits_refills_without_removing_excluded_probability_mass() {
        let catalog = standard_semantic_archetype_catalog();
        let ids: Vec<_> = catalog
            .definitions()
            .iter()
            .filter(|definition| definition.standard_tile_count == 1)
            .take(4)
            .map(|definition| definition.id)
            .collect();
        let mut counts = vec![0u16; catalog.len()];
        for id in &ids {
            counts[id.index()] = 1;
        }
        let supply = ExactSemanticSupply::from_exact_counts([20; 5], counts, 2).unwrap();

        assert_eq!(supply.unseen_tile_count(), 4);
        assert_eq!(supply.drawable_tile_count(), 2);
        assert_eq!(
            supply
                .refill_distribution(1)
                .unwrap()
                .probability_of_ordered(&[ids[0]])
                .unwrap(),
            ExactProbability::new(1, 4).unwrap()
        );
        assert!(supply.refill_distribution(2).is_ok());
        assert_eq!(
            supply.refill_distribution(3),
            Err(SemanticSupplyError::RefillExceedsSupply {
                requested: 3,
                available: 2,
            })
        );
    }

    #[test]
    fn exact_probabilities_normalize_for_one_and_multi_slot_refills() {
        let supply = ExactSemanticSupply::from_game(&game(19)).unwrap();
        let one = supply.refill_distribution(1).unwrap();
        assert_eq!(
            one.one_slot_probabilities()
                .iter()
                .map(|entry| entry.probability.numerator)
                .sum::<u64>(),
            u64::from(one.total_unseen())
        );
        assert!(
            one.one_slot_probabilities()
                .iter()
                .all(|entry| entry.probability.denominator
                    == u64::from(one.total_unseen())
                        / gcd(entry.probability.numerator, u64::from(one.total_unseen())))
        );

        let catalog = standard_semantic_archetype_catalog();
        let mut counts = vec![0u16; catalog.len()];
        let repeated = catalog
            .definitions()
            .iter()
            .find(|definition| definition.standard_tile_count >= 2)
            .unwrap()
            .id;
        let singles: Vec<_> = catalog
            .definitions()
            .iter()
            .filter(|definition| definition.id != repeated)
            .take(2)
            .map(|definition| definition.id)
            .collect();
        counts[repeated.index()] = 2;
        counts[singles[0].index()] = 1;
        counts[singles[1].index()] = 1;
        let fixture = ExactSemanticSupply::from_exact_counts([20; 5], counts, 4).unwrap();
        for slots in 1..=4 {
            let distribution = fixture.refill_distribution(slots).unwrap();
            let outcomes = distribution.enumerate_ordered_outcomes(64).unwrap();
            assert_eq!(
                outcomes
                    .iter()
                    .map(|outcome| outcome.ordered_weight)
                    .sum::<u64>(),
                distribution.ordered_denominator()
            );
            assert!(outcomes.iter().all(|outcome| {
                distribution
                    .probability_of_ordered(&outcome.archetypes)
                    .unwrap()
                    == outcome.probability
            }));
        }
    }

    #[test]
    fn conditional_and_unordered_refill_probabilities_are_exact() {
        let catalog = standard_semantic_archetype_catalog();
        let mut counts = vec![0u16; catalog.len()];
        let first = catalog
            .definitions()
            .iter()
            .find(|definition| definition.standard_tile_count >= 2)
            .unwrap()
            .id;
        let singles: Vec<_> = catalog
            .definitions()
            .iter()
            .filter(|definition| definition.id != first)
            .take(2)
            .map(|definition| definition.id)
            .collect();
        counts[first.index()] = 2;
        counts[singles[0].index()] = 1;
        counts[singles[1].index()] = 1;
        let supply = ExactSemanticSupply::from_exact_counts([20; 5], counts, 4).unwrap();
        let distribution = supply.refill_distribution(3).unwrap();
        let second = singles[0];

        assert_eq!(
            distribution
                .probability_of_ordered(&[first, first, second])
                .unwrap(),
            ExactProbability::new(2, 24).unwrap()
        );
        assert_eq!(
            distribution
                .probability_of_unordered(&[(first, 2), (second, 1)])
                .unwrap(),
            ExactProbability::new(1, 4).unwrap()
        );
        let conditional = distribution.conditional_after(&[first]).unwrap();
        assert_eq!(conditional.slot_count(), 2);
        assert_eq!(conditional.total_unseen(), 3);
        assert_eq!(conditional.archetype_counts[first.index()], 1);
    }

    #[test]
    fn exact_archetypes_separate_a_real_legacy_marginal_collision() {
        let catalog = standard_semantic_archetype_catalog();
        let left_tiles = [STANDARD_TILES[0], STANDARD_TILES[23]];
        let right_tiles = [STANDARD_TILES[2], STANDARD_TILES[20]];
        let supply_for = |tiles: [Tile; 2]| {
            let mut counts = vec![0u16; catalog.len()];
            for tile in tiles {
                let id = catalog.reference_for_tile(tile).unwrap().archetype_id;
                counts[id.index()] += 1;
            }
            ExactSemanticSupply::from_exact_counts([20; 5], counts, 2).unwrap()
        };
        let left = supply_for(left_tiles);
        let right = supply_for(right_tiles);

        assert_eq!(
            left.to_legacy_public_supply(),
            right.to_legacy_public_supply()
        );
        assert_ne!(left.archetype_counts(), right.archetype_counts());
        assert_ne!(left.canonical_hash(), right.canonical_hash());
    }

    #[test]
    fn every_two_tile_legacy_marginal_collision_is_exactly_separated() {
        let catalog = standard_semantic_archetype_catalog();
        let mut by_legacy_marginals =
            std::collections::BTreeMap::<[u8; 25], Vec<ExactSemanticSupply>>::new();
        for (left_index, left_tile) in STANDARD_TILES.into_iter().enumerate() {
            for right_tile in STANDARD_TILES.into_iter().skip(left_index + 1) {
                let mut counts = vec![0u16; catalog.len()];
                for tile in [left_tile, right_tile] {
                    let id = catalog.reference_for_tile(tile).unwrap().archetype_id;
                    counts[id.index()] += 1;
                }
                let supply = ExactSemanticSupply::from_exact_counts([20; 5], counts, 2).unwrap();
                let legacy = supply.to_legacy_public_supply();
                let mut signature = [0u8; 25];
                signature[..5].copy_from_slice(&legacy.unseen_tile_terrain_capacity);
                signature[5..10].copy_from_slice(&legacy.unseen_tile_wildlife_capacity);
                signature[10..15].copy_from_slice(&legacy.unseen_keystones_by_terrain);
                signature[15..].copy_from_slice(&legacy.unseen_dual_terrain_pairs);
                by_legacy_marginals
                    .entry(signature)
                    .or_default()
                    .push(supply);
            }
        }

        let mut exact_collisions_checked = 0usize;
        for supplies in by_legacy_marginals.values() {
            for left in 0..supplies.len() {
                for right in left + 1..supplies.len() {
                    if supplies[left].archetype_counts() == supplies[right].archetype_counts() {
                        continue;
                    }
                    exact_collisions_checked += 1;
                    assert_ne!(
                        supplies[left].canonical_bytes(),
                        supplies[right].canonical_bytes()
                    );
                    assert_ne!(
                        supplies[left].canonical_hash(),
                        supplies[right].canonical_hash()
                    );
                }
            }
        }
        assert!(
            exact_collisions_checked > 0,
            "the frozen official catalog should retain real legacy aliases"
        );
    }

    #[test]
    fn market_and_frontier_links_preserve_exact_rotation_semantics() {
        let state = game(23);
        let supply = ExactSemanticSupply::from_game(&state).unwrap();
        let links = supply.market_links(state.market()).unwrap();
        for slot in MarketSlot::ALL {
            let tile = state.market().tiles[slot.index()].unwrap();
            let link = links[slot.index()].unwrap();
            assert_eq!(link.slot, slot);
            let requirements = FrontierTerrainRequirements::new([
                Some(Terrain::Mountain),
                Some(Terrain::Forest),
                None,
                Some(Terrain::Prairie),
                None,
                Some(Terrain::River),
            ]);
            let compatibility = link
                .tile
                .frontier_compatibility(tile, requirements)
                .unwrap();
            for rotation in Rotation::ALL {
                let expected = requirements
                    .neighbor_facing_terrains
                    .into_iter()
                    .enumerate()
                    .filter(|(edge, terrain)| {
                        terrain
                            .is_some_and(|terrain| tile.terrain_on_edge(rotation, *edge) == terrain)
                    })
                    .count() as u8;
                assert_eq!(compatibility.matching_edges(rotation), expected);
            }
        }
    }

    #[test]
    fn frontier_compatibility_is_d6_covariant_for_every_standard_tile() {
        let catalog = standard_semantic_archetype_catalog();
        let requirements = FrontierTerrainRequirements::new([
            Some(Terrain::Mountain),
            Some(Terrain::Forest),
            None,
            Some(Terrain::Prairie),
            Some(Terrain::Wetland),
            None,
        ]);
        for tile in STANDARD_TILES {
            let reference = catalog.reference_for_tile(tile).unwrap();
            let source = reference
                .frontier_compatibility(tile, requirements)
                .unwrap();
            for transform in D6Transform::ALL {
                let transformed_requirements = requirements.transformed(transform).unwrap();
                let transformed = reference
                    .frontier_compatibility(tile, transformed_requirements)
                    .unwrap();
                for rotation in Rotation::ALL {
                    let transformed_rotation = transform.transform_tile_rotation(tile, rotation);
                    assert_eq!(
                        source.matching_edges(rotation),
                        transformed.matching_edges(transformed_rotation)
                    );
                }
                for direction in HexDirection::ALL {
                    assert_eq!(
                        transformed_requirements.neighbor_facing_terrains
                            [transform.transform_direction(direction).index()],
                        requirements.neighbor_facing_terrains[direction.index()]
                    );
                }
            }
        }
    }

    #[test]
    fn solo_states_fail_closed_without_public_discard_history() {
        let solo = GameState::new(
            GameConfig::solo(ScoringCards::AAAAA),
            GameSeed::from_u64(77),
        )
        .unwrap();
        assert_eq!(
            ExactSemanticSupply::from_game(&solo),
            Err(SemanticSupplyError::UnsupportedPublicMode(GameMode::Solo))
        );
    }
}
