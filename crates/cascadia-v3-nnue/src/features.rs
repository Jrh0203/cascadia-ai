use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    sync::OnceLock,
};

use cascadia_data::{
    OpportunityAssignment, OpportunityDemandKind, OpportunityGraphBuildContext, OpportunityGraphV1,
    PreparedOpportunityMatching, PreparedWildlifeMatching, PreparedWildlifeOpportunityDemands,
};
use cascadia_game::{
    Board, D6Transform, HexCoord, PlacedTile, PublicGameState, STANDARD_TILES, ScoringVariant,
    Terrain, Wildlife,
};
use serde::{Deserialize, Serialize};

use crate::{
    Result, V3Error,
    schema::{
        ALLOWED_WILDLIFE_BASE, BASE_FEATURE_ROWS, CORE_SPATIAL_FEATURE_ROWS, GLOBAL_BASE,
        GLOBAL_FEATURE_ROWS, HOT_CELL_COUNT, KEYSTONE_BASE, OPPORTUNITY_FEATURE_MAX,
        OPPORTUNITY_FEATURE_MIN, OVERFLOW_BASE, OVERFLOW_COORD_BINS, OVERFLOW_COORD_MAX,
        OVERFLOW_COORD_MIN, OVERFLOW_COUNT_BASE, OVERFLOW_SLOT_COUNT, OVERFLOW_SLOT_WIDTH,
        PLACED_WILDLIFE_BASE, TERRAIN_EDGE_BASE, TILE_PRESENCE_BASE, hot_coord, hot_index,
    },
};

const LOCATION_COUNT: u16 = HOT_CELL_COUNT as u16 + 1;
const OVERFLOW_LOCATION: u16 = HOT_CELL_COUNT as u16;
const ARCHETYPE_COUNT: u16 = 75;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ActiveFeature {
    pub index: u32,
    pub count: u16,
}

impl ActiveFeature {
    fn once(index: usize) -> Result<Self> {
        Ok(Self {
            index: u32::try_from(index)
                .map_err(|_| V3Error::InvalidFeature("feature index exceeds u32".to_owned()))?,
            count: 1,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OverflowEntity {
    pub q: i8,
    pub r: i8,
    pub directed_edge_terrains: [u8; 6],
    pub allowed_wildlife_bits: u8,
    pub placed_wildlife: Option<u8>,
    pub keystone: bool,
}

impl OverflowEntity {
    fn from_placed(coord: HexCoord, placed: &PlacedTile) -> Self {
        Self {
            q: coord.q,
            r: coord.r,
            directed_edge_terrains: std::array::from_fn(|edge| {
                placed.tile.terrain_on_edge(placed.rotation, edge) as u8
            }),
            allowed_wildlife_bits: placed.tile.wildlife.bits(),
            placed_wildlife: placed.wildlife.map(|wildlife| wildlife as u8),
            keystone: placed.tile.keystone,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BoardFeatureEncoding {
    pub active: Vec<ActiveFeature>,
    pub overflow: Vec<OverflowEntity>,
}

fn coordinate_bin(value: i8) -> Result<usize> {
    if !(OVERFLOW_COORD_MIN..=OVERFLOW_COORD_MAX).contains(&value) {
        return Err(V3Error::InvalidFeature(format!(
            "legal V3 overflow coordinate {value} is outside [{OVERFLOW_COORD_MIN}, {OVERFLOW_COORD_MAX}]"
        )));
    }
    Ok(usize::from((value - OVERFLOW_COORD_MIN) as u8))
}

fn consolidate(mut features: Vec<ActiveFeature>) -> Result<Vec<ActiveFeature>> {
    features.sort_unstable_by_key(|feature| feature.index);
    let mut consolidated = Vec::<ActiveFeature>::with_capacity(features.len());
    for feature in features {
        if feature.count == 0 {
            return Err(V3Error::InvalidFeature(
                "active feature has zero multiplicity".to_owned(),
            ));
        }
        if let Some(previous) = consolidated
            .last_mut()
            .filter(|row| row.index == feature.index)
        {
            previous.count = previous.count.checked_add(feature.count).ok_or_else(|| {
                V3Error::InvalidFeature("feature multiplicity overflow".to_owned())
            })?;
        } else {
            consolidated.push(feature);
        }
    }
    Ok(consolidated)
}

fn push_once(features: &mut Vec<ActiveFeature>, index: usize) -> Result<()> {
    features.push(ActiveFeature::once(index)?);
    Ok(())
}

pub fn encode_board_features(board: &Board) -> Result<BoardFeatureEncoding> {
    let mut active = Vec::with_capacity(board.tile_count() * 16 + 1);
    let mut overflow = board
        .placed_tiles()
        .filter(|(coord, _)| hot_index(*coord).is_none())
        .map(|(coord, placed)| OverflowEntity::from_placed(coord, placed))
        .collect::<Vec<_>>();
    overflow.sort_unstable_by_key(|entity| (entity.q, entity.r));
    if overflow.len() > OVERFLOW_SLOT_COUNT {
        return Err(V3Error::InvalidFeature(format!(
            "board has {} overflow entities; V3 supports {OVERFLOW_SLOT_COUNT}",
            overflow.len()
        )));
    }

    for (coord, placed) in board.placed_tiles() {
        let Some(cell) = hot_index(coord) else {
            continue;
        };
        push_once(&mut active, TILE_PRESENCE_BASE + cell)?;
        for edge in 0..6 {
            let terrain = placed.tile.terrain_on_edge(placed.rotation, edge) as usize;
            push_once(
                &mut active,
                TERRAIN_EDGE_BASE + (cell * 6 + edge) * 5 + terrain,
            )?;
        }
        for wildlife in placed.tile.wildlife.iter() {
            push_once(
                &mut active,
                ALLOWED_WILDLIFE_BASE + cell * 5 + wildlife as usize,
            )?;
        }
        if let Some(wildlife) = placed.wildlife {
            push_once(
                &mut active,
                PLACED_WILDLIFE_BASE + cell * 5 + wildlife as usize,
            )?;
        }
        if placed.tile.keystone {
            push_once(&mut active, KEYSTONE_BASE + cell)?;
        }
    }

    active.extend(encode_overflow_entities(&overflow)?);

    let active = consolidate(active)?;
    if active
        .iter()
        .any(|feature| feature.index as usize >= GLOBAL_BASE)
    {
        return Err(V3Error::InvalidFeature(
            "board encoder emitted a global feature".to_owned(),
        ));
    }
    Ok(BoardFeatureEncoding { active, overflow })
}

fn encode_overflow_entities(overflow: &[OverflowEntity]) -> Result<Vec<ActiveFeature>> {
    if overflow.len() > OVERFLOW_SLOT_COUNT
        || overflow
            .windows(2)
            .any(|pair| (pair[0].q, pair[0].r) >= (pair[1].q, pair[1].r))
    {
        return Err(V3Error::InvalidFeature(
            "overflow entities are over-capacity or not canonically sorted".to_owned(),
        ));
    }
    let mut active = Vec::with_capacity(overflow.len() * 16 + 1);
    for (slot, entity) in overflow.iter().enumerate() {
        let base = OVERFLOW_BASE + slot * OVERFLOW_SLOT_WIDTH;
        push_once(&mut active, base)?;
        push_once(&mut active, base + 1 + coordinate_bin(entity.q)?)?;
        push_once(
            &mut active,
            base + 1 + OVERFLOW_COORD_BINS + coordinate_bin(entity.r)?,
        )?;
        let edge_base = base + 1 + 2 * OVERFLOW_COORD_BINS;
        for (edge, terrain) in entity.directed_edge_terrains.iter().copied().enumerate() {
            push_once(&mut active, edge_base + edge * 5 + usize::from(terrain))?;
        }
        let allowed_base = edge_base + 6 * 5;
        for wildlife in 0..5 {
            if entity.allowed_wildlife_bits & (1 << wildlife) != 0 {
                push_once(&mut active, allowed_base + wildlife)?;
            }
        }
        let placed_base = allowed_base + 5;
        if let Some(wildlife) = entity.placed_wildlife {
            push_once(&mut active, placed_base + usize::from(wildlife))?;
        }
        if entity.keystone {
            push_once(&mut active, placed_base + 5)?;
        }
    }
    push_once(&mut active, OVERFLOW_COUNT_BASE + overflow.len())?;
    consolidate(active)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum OpportunityFeatureSpec {
    DemandLocation {
        relative_seat: u8,
        demand_kind: u8,
        subject: u8,
        location: u16,
    },
    MarketSynergy {
        relative_seat: u8,
        demand_kind: u8,
        subject: u8,
        location: u16,
        market_slot: u8,
    },
    Completion {
        relative_seat: u8,
        demand_kind: u8,
        subject: u8,
        location: u16,
        bucket: u8,
    },
    Archetype {
        relative_seat: u8,
        terrain: u8,
        archetype: u16,
    },
    AccessDelay {
        relative_seat: u8,
        demand_kind: u8,
        subject: u8,
        bucket: u8,
    },
    OpponentsBeforeAccess {
        relative_seat: u8,
        demand_kind: u8,
        subject: u8,
        bucket: u8,
    },
    MatchingEdges {
        relative_seat: u8,
        demand_kind: u8,
        subject: u8,
        bucket: u8,
    },
    RotationMask {
        relative_seat: u8,
        terrain: u8,
        mask: u8,
    },
    NatureActionability {
        relative_seat: u8,
        token_bucket: u8,
        wildlife: u8,
    },
    ContestedDraft {
        relative_seat: u8,
        market_slot: u8,
        opponents_before_access: u8,
    },
    PairedTileWildlife {
        relative_seat: u8,
        tile_slot: u8,
        wildlife_slot: u8,
        terrain: u8,
        wildlife: u8,
    },
    Deadline {
        relative_seat: u8,
        demand_kind: u8,
        subject: u8,
        deadline: u8,
        overflow: bool,
    },
    Availability {
        relative_seat: u8,
        demand_kind: u8,
        subject: u8,
        ratio_bucket: u8,
        delay_bucket: u8,
    },
}

/// Train-only sharing factors for the collision-free opportunity catalog.
///
/// Each inference row remains exact and independently addressable. During
/// training, active rows additionally activate these coordinate-free factors;
/// export adds their learned vectors into the corresponding inference row and
/// discards the factor table. This is the Stockfish-style virtual-feature
/// pattern: richer statistical sharing with zero serving-time indirection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum OpportunityTrainingFactorSpec {
    CoordinateFreeSemantic {
        family: u8,
        demand_kind: u8,
        subject: u8,
    },
    D6Orbit {
        family: u8,
        canonical_q: i8,
        canonical_r: i8,
    },
    MarketItem {
        family: u8,
        first: u8,
        second: u8,
        subject: u8,
    },
    Perspective {
        own_board: bool,
    },
    RelativeSeat {
        relative_seat: u8,
    },
    CompletionClass {
        family: u8,
        demand_kind: u8,
        subject: u8,
        class: u8,
    },
    CatalogArchetype {
        terrain: u8,
        archetype: u16,
    },
}

#[derive(Debug)]
pub struct FullOpportunitiesCatalog {
    specs: Vec<OpportunityFeatureSpec>,
    ids: HashMap<OpportunityFeatureSpec, u32>,
    variant_offsets: [u32; 13],
    checksum: String,
    training_factor_specs: Vec<OpportunityTrainingFactorSpec>,
    training_factor_offsets: Vec<u32>,
    training_factor_indices: Vec<u32>,
    training_factor_checksum: String,
}

fn opportunity_variant(spec: OpportunityFeatureSpec) -> usize {
    match spec {
        OpportunityFeatureSpec::DemandLocation { .. } => 0,
        OpportunityFeatureSpec::MarketSynergy { .. } => 1,
        OpportunityFeatureSpec::Completion { .. } => 2,
        OpportunityFeatureSpec::Archetype { .. } => 3,
        OpportunityFeatureSpec::AccessDelay { .. } => 4,
        OpportunityFeatureSpec::OpponentsBeforeAccess { .. } => 5,
        OpportunityFeatureSpec::MatchingEdges { .. } => 6,
        OpportunityFeatureSpec::RotationMask { .. } => 7,
        OpportunityFeatureSpec::NatureActionability { .. } => 8,
        OpportunityFeatureSpec::ContestedDraft { .. } => 9,
        OpportunityFeatureSpec::PairedTileWildlife { .. } => 10,
        OpportunityFeatureSpec::Deadline { .. } => 11,
        OpportunityFeatureSpec::Availability { .. } => 12,
    }
}

fn opportunity_family(spec: OpportunityFeatureSpec) -> u8 {
    match spec {
        OpportunityFeatureSpec::DemandLocation { .. } => 0,
        OpportunityFeatureSpec::MarketSynergy { .. } => 1,
        OpportunityFeatureSpec::Completion { .. } => 2,
        OpportunityFeatureSpec::Archetype { .. } => 3,
        OpportunityFeatureSpec::AccessDelay { .. } => 4,
        OpportunityFeatureSpec::OpponentsBeforeAccess { .. } => 5,
        OpportunityFeatureSpec::MatchingEdges { .. } => 6,
        OpportunityFeatureSpec::RotationMask { .. } => 7,
        OpportunityFeatureSpec::NatureActionability { .. } => 8,
        OpportunityFeatureSpec::ContestedDraft { .. } => 9,
        OpportunityFeatureSpec::PairedTileWildlife { .. } => 10,
        OpportunityFeatureSpec::Deadline { .. } => 11,
        OpportunityFeatureSpec::Availability { .. } => 12,
    }
}

fn relative_seat(spec: OpportunityFeatureSpec) -> u8 {
    match spec {
        OpportunityFeatureSpec::DemandLocation { relative_seat, .. }
        | OpportunityFeatureSpec::MarketSynergy { relative_seat, .. }
        | OpportunityFeatureSpec::Completion { relative_seat, .. }
        | OpportunityFeatureSpec::Archetype { relative_seat, .. }
        | OpportunityFeatureSpec::AccessDelay { relative_seat, .. }
        | OpportunityFeatureSpec::OpponentsBeforeAccess { relative_seat, .. }
        | OpportunityFeatureSpec::MatchingEdges { relative_seat, .. }
        | OpportunityFeatureSpec::RotationMask { relative_seat, .. }
        | OpportunityFeatureSpec::NatureActionability { relative_seat, .. }
        | OpportunityFeatureSpec::ContestedDraft { relative_seat, .. }
        | OpportunityFeatureSpec::PairedTileWildlife { relative_seat, .. }
        | OpportunityFeatureSpec::Deadline { relative_seat, .. }
        | OpportunityFeatureSpec::Availability { relative_seat, .. } => relative_seat,
    }
}

fn demand_semantics(spec: OpportunityFeatureSpec) -> (u8, u8) {
    match spec {
        OpportunityFeatureSpec::DemandLocation {
            demand_kind,
            subject,
            ..
        }
        | OpportunityFeatureSpec::MarketSynergy {
            demand_kind,
            subject,
            ..
        }
        | OpportunityFeatureSpec::Completion {
            demand_kind,
            subject,
            ..
        }
        | OpportunityFeatureSpec::AccessDelay {
            demand_kind,
            subject,
            ..
        }
        | OpportunityFeatureSpec::OpponentsBeforeAccess {
            demand_kind,
            subject,
            ..
        }
        | OpportunityFeatureSpec::MatchingEdges {
            demand_kind,
            subject,
            ..
        }
        | OpportunityFeatureSpec::Deadline {
            demand_kind,
            subject,
            ..
        }
        | OpportunityFeatureSpec::Availability {
            demand_kind,
            subject,
            ..
        } => (demand_kind, subject),
        OpportunityFeatureSpec::Archetype { terrain, .. }
        | OpportunityFeatureSpec::RotationMask { terrain, .. } => (u8::MAX, terrain),
        OpportunityFeatureSpec::NatureActionability { wildlife, .. } => (u8::MAX, wildlife),
        OpportunityFeatureSpec::PairedTileWildlife {
            terrain, wildlife, ..
        } => (terrain, wildlife),
        OpportunityFeatureSpec::ContestedDraft { .. } => (u8::MAX, u8::MAX),
    }
}

fn d6_orbit(location: u16) -> (i8, i8) {
    if location == OVERFLOW_LOCATION {
        return (i8::MAX, i8::MAX);
    }
    let coordinate = hot_coord(usize::from(location)).expect("catalog location is radius-7");
    D6Transform::ALL
        .into_iter()
        .map(|transform| {
            let transformed = transform
                .transform_coord(coordinate)
                .expect("radius-7 D6 transform is exact");
            (transformed.q, transformed.r)
        })
        .min()
        .expect("D6 contains transforms")
}

fn training_factors(spec: OpportunityFeatureSpec) -> Vec<OpportunityTrainingFactorSpec> {
    let family = opportunity_family(spec);
    let seat = relative_seat(spec);
    let (demand_kind, subject) = demand_semantics(spec);
    let mut factors = vec![
        OpportunityTrainingFactorSpec::CoordinateFreeSemantic {
            family,
            demand_kind,
            subject,
        },
        OpportunityTrainingFactorSpec::Perspective {
            own_board: seat == 0,
        },
        OpportunityTrainingFactorSpec::RelativeSeat {
            relative_seat: seat,
        },
    ];
    match spec {
        OpportunityFeatureSpec::DemandLocation { location, .. }
        | OpportunityFeatureSpec::MarketSynergy { location, .. }
        | OpportunityFeatureSpec::Completion { location, .. } => {
            let (canonical_q, canonical_r) = d6_orbit(location);
            factors.push(OpportunityTrainingFactorSpec::D6Orbit {
                family,
                canonical_q,
                canonical_r,
            });
        }
        _ => {}
    }
    match spec {
        OpportunityFeatureSpec::MarketSynergy {
            market_slot,
            subject,
            ..
        } => factors.push(OpportunityTrainingFactorSpec::MarketItem {
            family,
            first: market_slot,
            second: u8::MAX,
            subject,
        }),
        OpportunityFeatureSpec::ContestedDraft { market_slot, .. } => {
            factors.push(OpportunityTrainingFactorSpec::MarketItem {
                family,
                first: market_slot,
                second: u8::MAX,
                subject: u8::MAX,
            });
        }
        OpportunityFeatureSpec::PairedTileWildlife {
            tile_slot,
            wildlife_slot,
            wildlife,
            ..
        } => factors.push(OpportunityTrainingFactorSpec::MarketItem {
            family,
            first: tile_slot,
            second: wildlife_slot,
            subject: wildlife,
        }),
        OpportunityFeatureSpec::NatureActionability { wildlife, .. } => {
            factors.push(OpportunityTrainingFactorSpec::MarketItem {
                family,
                first: u8::MAX,
                second: u8::MAX,
                subject: wildlife,
            });
        }
        _ => {}
    }
    match spec {
        OpportunityFeatureSpec::Completion {
            demand_kind,
            subject,
            bucket,
            ..
        }
        | OpportunityFeatureSpec::AccessDelay {
            demand_kind,
            subject,
            bucket,
            ..
        }
        | OpportunityFeatureSpec::OpponentsBeforeAccess {
            demand_kind,
            subject,
            bucket,
            ..
        }
        | OpportunityFeatureSpec::MatchingEdges {
            demand_kind,
            subject,
            bucket,
            ..
        } => factors.push(OpportunityTrainingFactorSpec::CompletionClass {
            family,
            demand_kind,
            subject,
            class: bucket,
        }),
        OpportunityFeatureSpec::Deadline {
            demand_kind,
            subject,
            deadline,
            overflow,
            ..
        } => factors.push(OpportunityTrainingFactorSpec::CompletionClass {
            family,
            demand_kind,
            subject,
            class: deadline.saturating_add(if overflow { 21 } else { 0 }),
        }),
        OpportunityFeatureSpec::Availability {
            demand_kind,
            subject,
            ratio_bucket,
            delay_bucket,
            ..
        } => factors.push(OpportunityTrainingFactorSpec::CompletionClass {
            family,
            demand_kind,
            subject,
            class: ratio_bucket * 4 + delay_bucket,
        }),
        _ => {}
    }
    if let OpportunityFeatureSpec::Archetype {
        terrain, archetype, ..
    } = spec
    {
        factors.push(OpportunityTrainingFactorSpec::CatalogArchetype { terrain, archetype });
    }
    factors.sort_unstable();
    factors.dedup();
    factors
}

impl FullOpportunitiesCatalog {
    fn build() -> Self {
        let mut specs = BTreeSet::new();
        for relative_seat in 0..4 {
            for demand_kind in 0..2 {
                for subject in 0..5 {
                    for location in 0..LOCATION_COUNT {
                        specs.insert(OpportunityFeatureSpec::DemandLocation {
                            relative_seat,
                            demand_kind,
                            subject,
                            location,
                        });
                        for market_slot in 0..4 {
                            specs.insert(OpportunityFeatureSpec::MarketSynergy {
                                relative_seat,
                                demand_kind,
                                subject,
                                location,
                                market_slot,
                            });
                        }
                        for bucket in 0..6 {
                            specs.insert(OpportunityFeatureSpec::Completion {
                                relative_seat,
                                demand_kind,
                                subject,
                                location,
                                bucket,
                            });
                        }
                    }
                    for bucket in 0..4 {
                        specs.insert(OpportunityFeatureSpec::AccessDelay {
                            relative_seat,
                            demand_kind,
                            subject,
                            bucket,
                        });
                        specs.insert(OpportunityFeatureSpec::OpponentsBeforeAccess {
                            relative_seat,
                            demand_kind,
                            subject,
                            bucket,
                        });
                    }
                    for bucket in 0..7 {
                        specs.insert(OpportunityFeatureSpec::MatchingEdges {
                            relative_seat,
                            demand_kind,
                            subject,
                            bucket,
                        });
                    }
                    for deadline in 0..=20 {
                        for overflow in [false, true] {
                            specs.insert(OpportunityFeatureSpec::Deadline {
                                relative_seat,
                                demand_kind,
                                subject,
                                deadline,
                                overflow,
                            });
                        }
                    }
                    for ratio_bucket in 0..8 {
                        for delay_bucket in 0..4 {
                            specs.insert(OpportunityFeatureSpec::Availability {
                                relative_seat,
                                demand_kind,
                                subject,
                                ratio_bucket,
                                delay_bucket,
                            });
                        }
                    }
                }
            }
            for terrain in 0..5 {
                for archetype in 0..ARCHETYPE_COUNT {
                    specs.insert(OpportunityFeatureSpec::Archetype {
                        relative_seat,
                        terrain,
                        archetype,
                    });
                }
                for mask in 0..64 {
                    specs.insert(OpportunityFeatureSpec::RotationMask {
                        relative_seat,
                        terrain,
                        mask,
                    });
                }
            }
            for token_bucket in 0..24 {
                for wildlife in 0..5 {
                    specs.insert(OpportunityFeatureSpec::NatureActionability {
                        relative_seat,
                        token_bucket,
                        wildlife,
                    });
                }
            }
            for market_slot in 0..4 {
                for opponents_before_access in 0..4 {
                    specs.insert(OpportunityFeatureSpec::ContestedDraft {
                        relative_seat,
                        market_slot,
                        opponents_before_access,
                    });
                }
            }
            let valid_pairs = STANDARD_TILES
                .iter()
                .flat_map(|tile| {
                    Terrain::ALL.into_iter().filter_map(move |terrain| {
                        tile.contains_terrain(terrain)
                            .then_some((terrain, tile.wildlife))
                    })
                })
                .flat_map(|(terrain, wildlife_mask)| {
                    wildlife_mask
                        .iter()
                        .map(move |wildlife| (terrain as u8, wildlife as u8))
                })
                .collect::<BTreeSet<_>>();
            for tile_slot in 0..4 {
                for wildlife_slot in 0..4 {
                    for &(terrain, wildlife) in &valid_pairs {
                        specs.insert(OpportunityFeatureSpec::PairedTileWildlife {
                            relative_seat,
                            tile_slot,
                            wildlife_slot,
                            terrain,
                            wildlife,
                        });
                    }
                }
            }
        }
        let specs = specs.into_iter().collect::<Vec<_>>();
        assert!(
            (OPPORTUNITY_FEATURE_MIN..=OPPORTUNITY_FEATURE_MAX).contains(&specs.len()),
            "compiled FullOpportunities catalog has {} rows",
            specs.len()
        );
        let ids = specs
            .iter()
            .copied()
            .enumerate()
            .map(|(index, spec)| (spec, index as u32))
            .collect::<HashMap<_, _>>();
        let mut variant_offsets = [u32::MAX; 13];
        for (index, spec) in specs.iter().copied().enumerate() {
            let variant = opportunity_variant(spec);
            if variant_offsets[variant] == u32::MAX {
                variant_offsets[variant] = index as u32;
            }
        }
        assert!(variant_offsets.iter().all(|offset| *offset != u32::MAX));
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"cascadia-v3-full-opportunities-catalog-v1");
        for spec in &specs {
            let bytes = postcard::to_allocvec(spec).expect("catalog specs are serializable");
            hasher.update(&(bytes.len() as u32).to_le_bytes());
            hasher.update(&bytes);
        }
        let factor_sets = specs
            .iter()
            .copied()
            .map(training_factors)
            .collect::<Vec<_>>();
        let training_factor_specs = factor_sets
            .iter()
            .flatten()
            .copied()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        let training_factor_ids = training_factor_specs
            .iter()
            .copied()
            .enumerate()
            .map(|(index, factor)| (factor, index as u32))
            .collect::<BTreeMap<_, _>>();
        let mut training_factor_offsets = Vec::with_capacity(specs.len() + 1);
        let mut training_factor_indices = Vec::new();
        training_factor_offsets.push(0);
        for factors in factor_sets {
            training_factor_indices
                .extend(factors.iter().map(|factor| training_factor_ids[factor]));
            training_factor_offsets.push(training_factor_indices.len() as u32);
        }
        let mut factor_hasher = blake3::Hasher::new();
        factor_hasher.update(b"cascadia-v3-opportunity-training-factors-v1");
        for factor in &training_factor_specs {
            let bytes = postcard::to_allocvec(factor).expect("factor specs are serializable");
            factor_hasher.update(&(bytes.len() as u32).to_le_bytes());
            factor_hasher.update(&bytes);
        }
        for value in &training_factor_offsets {
            factor_hasher.update(&value.to_le_bytes());
        }
        for value in &training_factor_indices {
            factor_hasher.update(&value.to_le_bytes());
        }
        Self {
            specs,
            ids,
            variant_offsets,
            checksum: hasher.finalize().to_hex().to_string(),
            training_factor_specs,
            training_factor_offsets,
            training_factor_indices,
            training_factor_checksum: factor_hasher.finalize().to_hex().to_string(),
        }
    }

    pub fn global() -> &'static Self {
        static CATALOG: OnceLock<FullOpportunitiesCatalog> = OnceLock::new();
        CATALOG.get_or_init(Self::build)
    }

    pub fn len(&self) -> usize {
        self.specs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.specs.is_empty()
    }

    pub fn checksum(&self) -> &str {
        &self.checksum
    }

    pub fn id(&self, spec: OpportunityFeatureSpec) -> Result<u32> {
        let local = match spec {
            OpportunityFeatureSpec::DemandLocation {
                relative_seat,
                demand_kind,
                subject,
                location,
            } if relative_seat < 4
                && demand_kind < 2
                && subject < 5
                && location < LOCATION_COUNT =>
            {
                Some(
                    (((usize::from(relative_seat) * 2 + usize::from(demand_kind)) * 5
                        + usize::from(subject))
                        * usize::from(LOCATION_COUNT))
                        + usize::from(location),
                )
            }
            OpportunityFeatureSpec::MarketSynergy {
                relative_seat,
                demand_kind,
                subject,
                location,
                market_slot,
            } if relative_seat < 4
                && demand_kind < 2
                && subject < 5
                && location < LOCATION_COUNT
                && market_slot < 4 =>
            {
                Some(
                    ((((usize::from(relative_seat) * 2 + usize::from(demand_kind)) * 5
                        + usize::from(subject))
                        * usize::from(LOCATION_COUNT)
                        + usize::from(location))
                        * 4)
                        + usize::from(market_slot),
                )
            }
            OpportunityFeatureSpec::Completion {
                relative_seat,
                demand_kind,
                subject,
                location,
                bucket,
            } if relative_seat < 4
                && demand_kind < 2
                && subject < 5
                && location < LOCATION_COUNT
                && bucket < 6 =>
            {
                Some(
                    ((((usize::from(relative_seat) * 2 + usize::from(demand_kind)) * 5
                        + usize::from(subject))
                        * usize::from(LOCATION_COUNT)
                        + usize::from(location))
                        * 6)
                        + usize::from(bucket),
                )
            }
            OpportunityFeatureSpec::Archetype {
                relative_seat,
                terrain,
                archetype,
            } if relative_seat < 4 && terrain < 5 && archetype < ARCHETYPE_COUNT => Some(
                (usize::from(relative_seat) * 5 + usize::from(terrain))
                    * usize::from(ARCHETYPE_COUNT)
                    + usize::from(archetype),
            ),
            OpportunityFeatureSpec::AccessDelay {
                relative_seat,
                demand_kind,
                subject,
                bucket,
            }
            | OpportunityFeatureSpec::OpponentsBeforeAccess {
                relative_seat,
                demand_kind,
                subject,
                bucket,
            } if relative_seat < 4 && demand_kind < 2 && subject < 5 && bucket < 4 => Some(
                (((usize::from(relative_seat) * 2 + usize::from(demand_kind)) * 5
                    + usize::from(subject))
                    * 4)
                    + usize::from(bucket),
            ),
            OpportunityFeatureSpec::MatchingEdges {
                relative_seat,
                demand_kind,
                subject,
                bucket,
            } if relative_seat < 4 && demand_kind < 2 && subject < 5 && bucket < 7 => Some(
                (((usize::from(relative_seat) * 2 + usize::from(demand_kind)) * 5
                    + usize::from(subject))
                    * 7)
                    + usize::from(bucket),
            ),
            OpportunityFeatureSpec::RotationMask {
                relative_seat,
                terrain,
                mask,
            } if relative_seat < 4 && terrain < 5 && mask < 64 => Some(
                (usize::from(relative_seat) * 5 + usize::from(terrain)) * 64 + usize::from(mask),
            ),
            OpportunityFeatureSpec::NatureActionability {
                relative_seat,
                token_bucket,
                wildlife,
            } if relative_seat < 4 && token_bucket < 24 && wildlife < 5 => Some(
                (usize::from(relative_seat) * 24 + usize::from(token_bucket)) * 5
                    + usize::from(wildlife),
            ),
            OpportunityFeatureSpec::ContestedDraft {
                relative_seat,
                market_slot,
                opponents_before_access,
            } if relative_seat < 4 && market_slot < 4 && opponents_before_access < 4 => Some(
                (usize::from(relative_seat) * 4 + usize::from(market_slot)) * 4
                    + usize::from(opponents_before_access),
            ),
            OpportunityFeatureSpec::Deadline {
                relative_seat,
                demand_kind,
                subject,
                deadline,
                overflow,
            } if relative_seat < 4 && demand_kind < 2 && subject < 5 && deadline <= 20 => Some(
                ((((usize::from(relative_seat) * 2 + usize::from(demand_kind)) * 5
                    + usize::from(subject))
                    * 21
                    + usize::from(deadline))
                    * 2)
                    + usize::from(overflow),
            ),
            OpportunityFeatureSpec::Availability {
                relative_seat,
                demand_kind,
                subject,
                ratio_bucket,
                delay_bucket,
            } if relative_seat < 4
                && demand_kind < 2
                && subject < 5
                && ratio_bucket < 8
                && delay_bucket < 4 =>
            {
                Some(
                    ((((usize::from(relative_seat) * 2 + usize::from(demand_kind)) * 5
                        + usize::from(subject))
                        * 8
                        + usize::from(ratio_bucket))
                        * 4)
                        + usize::from(delay_bucket),
                )
            }
            OpportunityFeatureSpec::PairedTileWildlife { .. } => {
                return self.ids.get(&spec).copied().ok_or_else(|| {
                    V3Error::InvalidFeature(format!(
                        "opportunity feature is outside the catalog: {spec:?}"
                    ))
                });
            }
            _ => None,
        };
        let direct = local.and_then(|local| {
            let index = self.variant_offsets[opportunity_variant(spec)] as usize + local;
            (self.specs.get(index) == Some(&spec)).then_some(index as u32)
        });
        direct
            .or_else(|| self.ids.get(&spec).copied())
            .ok_or_else(|| {
                V3Error::InvalidFeature(format!(
                    "opportunity feature is outside the catalog: {spec:?}"
                ))
            })
    }

    pub fn spec(&self, index: u32) -> Option<OpportunityFeatureSpec> {
        self.specs.get(index as usize).copied()
    }

    pub fn training_factor_len(&self) -> usize {
        self.training_factor_specs.len()
    }

    pub fn training_factor_checksum(&self) -> &str {
        &self.training_factor_checksum
    }

    pub fn training_factor_offsets(&self) -> &[u32] {
        &self.training_factor_offsets
    }

    pub fn training_factor_indices(&self) -> &[u32] {
        &self.training_factor_indices
    }

    pub fn training_factor_spec(&self, index: u32) -> Option<OpportunityTrainingFactorSpec> {
        self.training_factor_specs.get(index as usize).copied()
    }

    pub fn training_factors_for_row(&self, index: u32) -> Option<&[u32]> {
        let row = usize::try_from(index).ok()?;
        let start = *self.training_factor_offsets.get(row)? as usize;
        let end = *self.training_factor_offsets.get(row + 1)? as usize;
        self.training_factor_indices.get(start..end)
    }
}

fn opportunity_location(coord: HexCoord) -> u16 {
    hot_index(coord).map_or(OVERFLOW_LOCATION, |index| index as u16)
}

fn opportunity_features(
    state: &PublicGameState,
    absolute_seat: usize,
    relative_seat: u8,
) -> Result<Vec<ActiveFeature>> {
    let context = OpportunityGraphBuildContext::new(state, absolute_seat)?;
    let (graph, assignments) = context.build_and_solve_for_board(&state.boards()[absolute_seat])?;
    compile_opportunity_features(
        state,
        &state.boards()[absolute_seat],
        relative_seat,
        &graph,
        &assignments,
        true,
    )
}

fn opportunity_features_with_context(
    context: &OpportunityGraphBuildContext,
    board: &Board,
    relative_seat: u8,
) -> Result<Vec<ActiveFeature>> {
    static PROFILE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let profile =
        *PROFILE.get_or_init(|| std::env::var_os("CASCADIA_V3_PROFILE_OPPORTUNITY").is_some());
    let started = std::time::Instant::now();
    let (graph, assignments) = context.build_and_solve_for_board(board)?;
    let graph_seconds = started.elapsed().as_secs_f64();
    let matching_seconds = 0.0;
    let started = std::time::Instant::now();
    let features = compile_opportunity_features(
        context.state(),
        board,
        relative_seat,
        &graph,
        &assignments,
        true,
    )?;
    if profile {
        static PRINTED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        if PRINTED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 8 {
            eprintln!(
                "V3_OPPORTUNITY_PROFILE {}",
                serde_json::json!({
                    "demands": graph.demands.len(),
                    "supplies": graph.supplies.len(),
                    "edges": graph.edges.len(),
                    "assignments": assignments.len(),
                    "features": features.len(),
                    "graph_seconds": graph_seconds,
                    "matching_seconds": matching_seconds,
                    "compile_seconds": started.elapsed().as_secs_f64(),
                })
            );
        }
    }
    Ok(features)
}

fn compile_opportunity_features(
    state: &PublicGameState,
    board: &Board,
    relative_seat: u8,
    graph: &OpportunityGraphV1,
    assignments: &[OpportunityAssignment],
    include_market_pairs: bool,
) -> Result<Vec<ActiveFeature>> {
    let catalog = FullOpportunitiesCatalog::global();
    let mut active = Vec::with_capacity(graph.demands.len() * 3 + graph.edges.len() * 8 + 32);

    for assignment in assignments {
        let demand = graph
            .demands
            .binary_search_by_key(&assignment.demand, |demand| demand.id)
            .ok()
            .and_then(|index| graph.demands.get(index))
            .ok_or_else(|| {
                V3Error::InvalidFeature("matching references a missing demand".to_owned())
            })?;
        let kind = demand.id.kind as u8;
        let subject = demand.id.subject;
        let location = opportunity_location(demand.id.coord);
        active.push(ActiveFeature {
            index: catalog.id(OpportunityFeatureSpec::DemandLocation {
                relative_seat,
                demand_kind: kind,
                subject,
                location,
            })?,
            count: 1,
        });
        active.push(ActiveFeature {
            index: catalog.id(OpportunityFeatureSpec::Deadline {
                relative_seat,
                demand_kind: kind,
                subject,
                deadline: demand.deadline_turns.min(20),
                overflow: location == OVERFLOW_LOCATION,
            })?,
            count: 1,
        });
        if demand.id.kind == OpportunityDemandKind::WildlifePlacement {
            active.push(ActiveFeature {
                index: catalog.id(OpportunityFeatureSpec::NatureActionability {
                    relative_seat,
                    token_bucket: board.nature_tokens().min(23),
                    wildlife: subject,
                })?,
                count: 1,
            });
        }
    }

    for assignment in assignments {
        let edge_id = cascadia_data::OpportunityEdgeId {
            demand: assignment.demand,
            supply: assignment.supply,
        };
        let edge = graph
            .edges
            .binary_search_by_key(&edge_id, |edge| edge.id)
            .ok()
            .and_then(|index| graph.edges.get(index))
            .ok_or_else(|| {
                V3Error::InvalidFeature("matching references a missing edge".to_owned())
            })?;
        let demand = edge.id.demand;
        let supply = graph
            .supplies
            .binary_search_by_key(&edge.id.supply, |supply| supply.id)
            .ok()
            .and_then(|index| graph.supplies.get(index))
            .ok_or_else(|| {
                V3Error::InvalidFeature("matching references a missing supply".to_owned())
            })?;
        let kind = demand.kind as u8;
        let subject = demand.subject;
        let location = opportunity_location(demand.coord);
        if let Some(slot) = supply.market_slot {
            active.push(ActiveFeature {
                index: catalog.id(OpportunityFeatureSpec::MarketSynergy {
                    relative_seat,
                    demand_kind: kind,
                    subject,
                    location,
                    market_slot: slot.index() as u8,
                })?,
                count: 1,
            });
            active.push(ActiveFeature {
                index: catalog.id(OpportunityFeatureSpec::ContestedDraft {
                    relative_seat,
                    market_slot: slot.index() as u8,
                    opponents_before_access: supply.opponents_before_access.min(3),
                })?,
                count: 1,
            });
        }
        active.push(ActiveFeature {
            index: catalog.id(OpportunityFeatureSpec::Completion {
                relative_seat,
                demand_kind: kind,
                subject,
                location,
                bucket: edge.exact_completion_delta.saturating_sub(1).min(5) as u8,
            })?,
            count: 1,
        });
        active.push(ActiveFeature {
            index: catalog.id(OpportunityFeatureSpec::AccessDelay {
                relative_seat,
                demand_kind: kind,
                subject,
                bucket: supply.access_delay_turns.min(3),
            })?,
            count: 1,
        });
        active.push(ActiveFeature {
            index: catalog.id(OpportunityFeatureSpec::OpponentsBeforeAccess {
                relative_seat,
                demand_kind: kind,
                subject,
                bucket: supply.opponents_before_access.min(3),
            })?,
            count: 1,
        });
        active.push(ActiveFeature {
            index: catalog.id(OpportunityFeatureSpec::MatchingEdges {
                relative_seat,
                demand_kind: kind,
                subject,
                bucket: edge.best_matching_edges.min(6),
            })?,
            count: 1,
        });
        let ratio_bucket = ((u32::from(supply.availability_numerator) * 8)
            / u32::from(supply.availability_denominator))
        .min(7) as u8;
        active.push(ActiveFeature {
            index: catalog.id(OpportunityFeatureSpec::Availability {
                relative_seat,
                demand_kind: kind,
                subject,
                ratio_bucket,
                delay_bucket: supply.access_delay_turns.min(3),
            })?,
            count: 1,
        });
        if let Some(archetype) = supply.archetype_id {
            active.push(ActiveFeature {
                index: catalog.id(OpportunityFeatureSpec::Archetype {
                    relative_seat,
                    terrain: subject,
                    archetype: archetype.code(),
                })?,
                count: 1,
            });
            active.push(ActiveFeature {
                index: catalog.id(OpportunityFeatureSpec::RotationMask {
                    relative_seat,
                    terrain: subject,
                    mask: edge.compatible_rotation_mask,
                })?,
                count: 1,
            });
        }
    }

    if include_market_pairs {
        for tile_slot in 0..4 {
            let Some(tile) = state.market().tiles[tile_slot] else {
                continue;
            };
            for wildlife_slot in 0..4 {
                let Some(wildlife) = state.market().wildlife[wildlife_slot] else {
                    continue;
                };
                if !tile.wildlife.contains(wildlife) {
                    continue;
                }
                for terrain in Terrain::ALL {
                    if tile.contains_terrain(terrain) {
                        active.push(ActiveFeature {
                            index: catalog.id(OpportunityFeatureSpec::PairedTileWildlife {
                                relative_seat,
                                tile_slot: tile_slot as u8,
                                wildlife_slot: wildlife_slot as u8,
                                terrain: terrain as u8,
                                wildlife: wildlife as u8,
                            })?,
                            count: 1,
                        });
                    }
                }
            }
        }
    }
    consolidate(active)
}

fn scoring_variant_index(variant: ScoringVariant) -> usize {
    match variant {
        ScoringVariant::A => 0,
        ScoringVariant::B => 1,
        ScoringVariant::C => 2,
        ScoringVariant::D => 3,
    }
}

fn global_features(state: &PublicGameState, focal_seat: usize) -> Result<Vec<ActiveFeature>> {
    if state.boards().len() != 4 || focal_seat >= 4 {
        return Err(V3Error::InvalidFeature(
            "V3 requires a four-player public state and valid focal seat".to_owned(),
        ));
    }
    let mut offset = 0usize;
    let phase_base = offset;
    offset += 21;
    let focal_seat_base = offset;
    offset += 4;
    let player_count_base = offset;
    offset += 1;
    let tile_count_base = offset;
    offset += 4 * 24;
    let nature_base = offset;
    offset += 4 * 24;
    let wildlife_count_base = offset;
    offset += 4 * 5 * 21;
    let habitat_size_base = offset;
    offset += 4 * 5 * 24;
    let market_tile_presence_base = offset;
    offset += 4 * 2;
    let market_wildlife_presence_base = offset;
    offset += 4 * 2;
    let market_terrain_a_base = offset;
    offset += 4 * 6;
    let market_terrain_b_base = offset;
    offset += 4 * 6;
    let market_allowed_base = offset;
    offset += 4 * 5;
    let market_wildlife_base = offset;
    offset += 4 * 6;
    let market_keystone_base = offset;
    offset += 4 * 2;
    let scoring_base = offset;
    offset += 5 * 4;
    let turns_remaining_base = offset;
    offset += 4 * 21;
    let compatibility_base = offset;
    offset += 4 * 2;
    let phase_nature_base = offset;
    offset += 21 * 16;
    let habitat_bonus_base = offset;
    offset += 2;
    let independent_base = offset;
    offset += 2;
    let game_over_feature = offset;
    offset += 1;
    let three_of_kind_base = offset;
    offset += 6;
    let current_relative_base = offset;
    offset += 4;
    let market_tile_complete_base = offset;
    offset += 2;
    let market_wildlife_complete_base = offset;
    offset += 2;
    let paid_wipe_available_base = offset;
    offset += 2;
    debug_assert_eq!(offset, GLOBAL_FEATURE_ROWS);

    let own_turns = state.boards()[focal_seat]
        .tile_count()
        .saturating_sub(3)
        .min(20);
    let mut active = Vec::with_capacity(128);
    let mut push = |local: usize| push_once(&mut active, GLOBAL_BASE + local);
    push(phase_base + own_turns)?;
    push(focal_seat_base + focal_seat)?;
    push(player_count_base)?;

    for relative in 0..4 {
        let absolute = (focal_seat + relative) % 4;
        let board = &state.boards()[absolute];
        push(tile_count_base + relative * 24 + board.tile_count().min(23))?;
        push(nature_base + relative * 24 + usize::from(board.nature_tokens().min(23)))?;
        for wildlife in Wildlife::ALL {
            let count = board.wildlife_positions(wildlife).len().min(20);
            push(wildlife_count_base + (relative * 5 + wildlife as usize) * 21 + count)?;
        }
        for terrain in Terrain::ALL {
            let size = usize::from(board.largest_habitat(terrain).min(23));
            push(habitat_size_base + (relative * 5 + terrain as usize) * 24 + size)?;
        }
        let remaining = usize::from(state.turns_remaining_for_player(absolute).min(20));
        push(turns_remaining_base + relative * 21 + remaining)?;
    }

    for slot in 0..4 {
        let tile = state.market().tiles[slot];
        let wildlife = state.market().wildlife[slot];
        push(market_tile_presence_base + slot * 2 + usize::from(tile.is_some()))?;
        push(market_wildlife_presence_base + slot * 2 + usize::from(wildlife.is_some()))?;
        push(market_terrain_a_base + slot * 6 + tile.map_or(5, |value| value.terrain_a as usize))?;
        push(
            market_terrain_b_base
                + slot * 6
                + tile
                    .and_then(|value| value.terrain_b)
                    .map_or(5, |terrain| terrain as usize),
        )?;
        if let Some(tile) = tile {
            for allowed in tile.wildlife.iter() {
                push(market_allowed_base + slot * 5 + allowed as usize)?;
            }
        }
        push(market_wildlife_base + slot * 6 + wildlife.map_or(5, |value| value as usize))?;
        push(
            market_keystone_base + slot * 2 + usize::from(tile.is_some_and(|value| value.keystone)),
        )?;
        let compatible = tile
            .zip(wildlife)
            .is_some_and(|(tile, wildlife)| tile.wildlife.contains(wildlife));
        push(compatibility_base + slot * 2 + usize::from(compatible))?;
    }

    let cards = state.config().scoring_cards;
    for (index, card) in [cards.bear, cards.elk, cards.salmon, cards.hawk, cards.fox]
        .into_iter()
        .enumerate()
    {
        push(scoring_base + index * 4 + scoring_variant_index(card))?;
    }
    push(
        phase_nature_base
            + own_turns * 16
            + usize::from(state.boards()[focal_seat].nature_tokens().min(15)),
    )?;
    push(habitat_bonus_base + usize::from(state.config().habitat_bonuses))?;
    push(independent_base + usize::from(state.boards()[focal_seat].nature_tokens() > 0))?;
    if state.is_game_over() {
        push(game_over_feature)?;
    }
    push(
        three_of_kind_base
            + state
                .market()
                .three_of_a_kind()
                .map_or(5, |wildlife| wildlife as usize),
    )?;
    let current_relative = (state.current_player() + 4 - focal_seat) % 4;
    push(current_relative_base + current_relative)?;
    push(
        market_tile_complete_base + usize::from(state.market().tiles.iter().all(Option::is_some)),
    )?;
    push(
        market_wildlife_complete_base
            + usize::from(state.market().wildlife.iter().all(Option::is_some)),
    )?;
    let paid_wipe_available = state.boards()[focal_seat].nature_tokens() > 0
        && state.market().wildlife.iter().any(Option::is_some);
    push(paid_wipe_available_base + usize::from(paid_wipe_available))?;
    consolidate(active)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct V3FeatureSet {
    pub own_base: Vec<ActiveFeature>,
    pub field_base: Vec<ActiveFeature>,
    pub own_opportunities: Vec<ActiveFeature>,
    pub field_opportunities: Vec<ActiveFeature>,
    pub overflow_entities: Vec<Vec<OverflowEntity>>,
    pub phase_bucket: u8,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct V3OwnFeatureSet {
    pub own_base: Vec<ActiveFeature>,
    pub own_opportunities: Vec<ActiveFeature>,
    pub phase_bucket: u8,
}

fn replace_active_once(
    features: &mut Vec<ActiveFeature>,
    old_index: usize,
    new_index: usize,
) -> Result<()> {
    if old_index == new_index {
        return Ok(());
    }
    let old = u32::try_from(old_index)
        .map_err(|_| V3Error::InvalidFeature("feature index exceeds u32".to_owned()))?;
    let position = features
        .binary_search_by_key(&old, |feature| feature.index)
        .map_err(|_| {
            let nearby = features
                .iter()
                .filter(|feature| {
                    feature.index.abs_diff(old) <= 2
                        || feature.index.abs_diff(u32::try_from(new_index).unwrap_or(u32::MAX)) <= 2
                })
                .cloned()
                .collect::<Vec<_>>();
            V3Error::InvalidFeature(format!(
                "prepared sibling base is missing replaced row {old_index} -> {new_index}; nearby={nearby:?}"
            ))
        })?;
    if features[position].count != 1 {
        return Err(V3Error::InvalidFeature(
            "prepared sibling replacement row is not binary".to_owned(),
        ));
    }
    features.remove(position);
    let replacement = ActiveFeature::once(new_index)?;
    let insertion = match features.binary_search_by_key(&replacement.index, |feature| feature.index)
    {
        Ok(_) => {
            return Err(V3Error::InvalidFeature(
                "sibling replacement duplicates an active row".to_owned(),
            ));
        }
        Err(insertion) => insertion,
    };
    features.insert(insertion, replacement);
    Ok(())
}

fn wildlife_sibling_base_features(
    prepared: &[ActiveFeature],
    before: &Board,
    after: &Board,
    state: &PublicGameState,
    wildlife: Wildlife,
    coord: HexCoord,
) -> Result<Vec<ActiveFeature>> {
    let cell = hot_index(coord);
    let mut features = if cell.is_some() {
        prepared.to_vec()
    } else {
        let encoded = encode_board_features(after)?;
        let mut full = encoded.active;
        full.extend(
            prepared
                .iter()
                .filter(|feature| feature.index as usize >= GLOBAL_BASE)
                .cloned(),
        );
        consolidate(full)?
    };

    const NATURE_BASE_LOCAL: usize = 122;
    const WILDLIFE_COUNT_BASE_LOCAL: usize = 218;
    const PHASE_NATURE_BASE_LOCAL: usize = 1346;
    const INDEPENDENT_BASE_LOCAL: usize = 1684;
    const PAID_WIPE_AVAILABLE_BASE_LOCAL: usize = 1701;

    let before_count = before.wildlife_positions(wildlife).len().min(20);
    let after_count = after.wildlife_positions(wildlife).len().min(20);
    replace_active_once(
        &mut features,
        GLOBAL_BASE + WILDLIFE_COUNT_BASE_LOCAL + wildlife as usize * 21 + before_count,
        GLOBAL_BASE + WILDLIFE_COUNT_BASE_LOCAL + wildlife as usize * 21 + after_count,
    )?;

    let before_tokens = usize::from(before.nature_tokens().min(23));
    let after_tokens = usize::from(after.nature_tokens().min(23));
    replace_active_once(
        &mut features,
        GLOBAL_BASE + NATURE_BASE_LOCAL + before_tokens,
        GLOBAL_BASE + NATURE_BASE_LOCAL + after_tokens,
    )?;
    let own_turns = after.tile_count().saturating_sub(3).min(20);
    replace_active_once(
        &mut features,
        GLOBAL_BASE
            + PHASE_NATURE_BASE_LOCAL
            + own_turns * 16
            + usize::from(before.nature_tokens().min(15)),
        GLOBAL_BASE
            + PHASE_NATURE_BASE_LOCAL
            + own_turns * 16
            + usize::from(after.nature_tokens().min(15)),
    )?;
    replace_active_once(
        &mut features,
        GLOBAL_BASE + INDEPENDENT_BASE_LOCAL + usize::from(before.nature_tokens() > 0),
        GLOBAL_BASE + INDEPENDENT_BASE_LOCAL + usize::from(after.nature_tokens() > 0),
    )?;
    let market_has_wildlife = state.market().wildlife.iter().any(Option::is_some);
    replace_active_once(
        &mut features,
        GLOBAL_BASE
            + PAID_WIPE_AVAILABLE_BASE_LOCAL
            + usize::from(before.nature_tokens() > 0 && market_has_wildlife),
        GLOBAL_BASE
            + PAID_WIPE_AVAILABLE_BASE_LOCAL
            + usize::from(after.nature_tokens() > 0 && market_has_wildlife),
    )?;
    if let Some(cell) = cell {
        features.push(ActiveFeature::once(
            PLACED_WILDLIFE_BASE + cell * 5 + wildlife as usize,
        )?);
    }
    consolidate(features)
}

#[derive(Debug, Clone)]
pub struct V3FeatureContext {
    focal_seat: usize,
    opponent_hashes: [[u8; 32]; 3],
    field_base: Vec<ActiveFeature>,
    opponent_overflow: Vec<Vec<OverflowEntity>>,
}

/// The immutable opportunity artifacts shared by one tile-sibling candidate
/// group. Grouping these references makes the cache boundary explicit and
/// prevents individual encoders from drifting into different reuse contracts.
#[derive(Clone, Copy)]
pub struct PreparedOpportunityEvaluation<'a> {
    context: &'a OpportunityGraphBuildContext,
    habitat: &'a OpportunityGraphV1,
    wildlife: &'a PreparedWildlifeOpportunityDemands,
    habitat_matching: &'a PreparedOpportunityMatching,
    wildlife_matching: &'a PreparedWildlifeMatching,
}

impl<'a> PreparedOpportunityEvaluation<'a> {
    pub fn new(
        context: &'a OpportunityGraphBuildContext,
        habitat: &'a OpportunityGraphV1,
        wildlife: &'a PreparedWildlifeOpportunityDemands,
        habitat_matching: &'a PreparedOpportunityMatching,
        wildlife_matching: &'a PreparedWildlifeMatching,
    ) -> Self {
        Self {
            context,
            habitat,
            wildlife,
            habitat_matching,
            wildlife_matching,
        }
    }
}

impl V3FeatureContext {
    pub fn new(state: &PublicGameState, focal_seat: usize) -> Result<Self> {
        if state.boards().len() != 4 || focal_seat >= 4 {
            return Err(V3Error::InvalidFeature(
                "V3 feature context requires four boards".to_owned(),
            ));
        }
        let mut opponent_hashes = [[0u8; 32]; 3];
        let mut field_rows = Vec::new();
        let mut opponent_overflow = Vec::with_capacity(3);
        for relative in 1..4 {
            let absolute = (focal_seat + relative) % 4;
            opponent_hashes[relative - 1] = *state.boards()[absolute].canonical_hash().as_bytes();
            let encoded = encode_board_features(&state.boards()[absolute])?;
            field_rows.extend(encoded.active);
            opponent_overflow.push(encoded.overflow);
        }
        Ok(Self {
            focal_seat,
            opponent_hashes,
            field_base: consolidate(field_rows)?,
            opponent_overflow,
        })
    }

    pub fn encode_afterstate(&self, state: &PublicGameState) -> Result<V3FeatureSet> {
        let field_opportunities = self.field_opportunities(state)?;
        self.encode_afterstate_with_field(state, field_opportunities)
    }

    pub fn field_opportunities(&self, state: &PublicGameState) -> Result<Vec<ActiveFeature>> {
        self.validate_opponents(state)?;
        let mut rows = Vec::new();
        for relative in 1..4 {
            let absolute = (self.focal_seat + relative) % 4;
            rows.extend(opportunity_features(state, absolute, relative as u8)?);
        }
        consolidate(rows)
    }

    pub fn encode_afterstate_with_field(
        &self,
        state: &PublicGameState,
        field_opportunities: Vec<ActiveFeature>,
    ) -> Result<V3FeatureSet> {
        self.validate_opponents(state)?;
        if field_opportunities.iter().any(|row| {
            row.index as usize >= FullOpportunitiesCatalog::global().len() || row.count == 0
        }) {
            return Err(V3Error::InvalidFeature(
                "cached field opportunities are invalid".to_owned(),
            ));
        }
        let own_board = encode_board_features(&state.boards()[self.focal_seat])?;
        let mut own_base = own_board.active.clone();
        own_base.extend(global_features(state, self.focal_seat)?);
        let completed = state.boards()[self.focal_seat]
            .tile_count()
            .saturating_sub(3)
            .min(20);
        let mut overflow_entities = Vec::with_capacity(4);
        overflow_entities.push(own_board.overflow);
        overflow_entities.extend(self.opponent_overflow.iter().cloned());
        let features = V3FeatureSet {
            own_base: consolidate(own_base)?,
            field_base: self.field_base.clone(),
            own_opportunities: opportunity_features(state, self.focal_seat, 0)?,
            field_opportunities,
            overflow_entities,
            phase_bucket: ((8 * completed) / 20).min(7) as u8,
        };
        features.validate()?;
        Ok(features)
    }

    pub fn encode_afterstate_board_with_field(
        &self,
        state: &PublicGameState,
        own_board: &Board,
        own_opportunity_context: &OpportunityGraphBuildContext,
        field_opportunities: Vec<ActiveFeature>,
    ) -> Result<V3FeatureSet> {
        self.validate_opponents(state)?;
        if field_opportunities.iter().any(|row| {
            row.index as usize >= FullOpportunitiesCatalog::global().len() || row.count == 0
        }) {
            return Err(V3Error::InvalidFeature(
                "cached field opportunities are invalid".to_owned(),
            ));
        }
        let completed = own_board.tile_count().saturating_sub(3).min(20);
        let encoded_own_board = encode_board_features(own_board)?;
        let mut own_base = encoded_own_board.active.clone();
        own_base.extend(global_features(state, self.focal_seat)?);
        let mut overflow_entities = Vec::with_capacity(4);
        overflow_entities.push(encoded_own_board.overflow);
        overflow_entities.extend(self.opponent_overflow.iter().cloned());
        let features = V3FeatureSet {
            own_base: consolidate(own_base)?,
            field_base: self.field_base.clone(),
            own_opportunities: opportunity_features_with_context(
                own_opportunity_context,
                own_board,
                0,
            )?,
            field_opportunities,
            overflow_entities,
            phase_bucket: ((8 * completed) / 20).min(7) as u8,
        };
        features.validate()?;
        Ok(features)
    }

    /// Build the placement-only habitat graph once for every tile sibling
    /// group. The shared matcher is intentionally not run until the exact
    /// wildlife afterstate is known.
    pub fn habitat_opportunity_graph_with_context(
        &self,
        board: &Board,
        context: &OpportunityGraphBuildContext,
    ) -> Result<OpportunityGraphV1> {
        Ok(context.build_for_board_kind(board, OpportunityDemandKind::HabitatFrontier)?)
    }

    /// Encode an exact wildlife sibling using a cached habitat graph. The
    /// original shared-capacity matcher runs over the merged graph, preserving
    /// bit-identical opportunity features and ranking semantics.
    pub fn encode_afterstate_board_with_cached_habitat(
        &self,
        state: &PublicGameState,
        own_board: &Board,
        prepared: PreparedOpportunityEvaluation<'_>,
        placed_wildlife: Option<(Wildlife, HexCoord)>,
        field_opportunities: Vec<ActiveFeature>,
    ) -> Result<V3FeatureSet> {
        debug_assert!(self.validate_opponents(state).is_ok());
        let completed = own_board.tile_count().saturating_sub(3).min(20);
        let encoded_own_board = encode_board_features(own_board)?;
        let mut own_base = encoded_own_board.active.clone();
        own_base.extend(global_features(state, self.focal_seat)?);
        let (graph, assignments) = prepared.context.build_and_solve_with_matching_frontiers(
            own_board,
            prepared.habitat,
            prepared.wildlife,
            prepared.habitat_matching,
            prepared.wildlife_matching,
            placed_wildlife,
        )?;
        let own_opportunities = compile_opportunity_features(
            prepared.context.state(),
            own_board,
            0,
            &graph,
            &assignments,
            true,
        )?;
        let mut overflow_entities = Vec::with_capacity(4);
        overflow_entities.push(encoded_own_board.overflow);
        overflow_entities.extend(self.opponent_overflow.iter().cloned());
        let features = V3FeatureSet {
            own_base: consolidate(own_base)?,
            field_base: self.field_base.clone(),
            own_opportunities,
            field_opportunities,
            overflow_entities,
            phase_bucket: ((8 * completed) / 20).min(7) as u8,
        };
        debug_assert!(features.validate().is_ok());
        Ok(features)
    }

    /// Candidate-only own-side encoding for a wildlife sibling. Opponent
    /// fields are already represented by the prepared field accumulator.
    pub fn encode_wildlife_sibling_own_with_cached_habitat(
        &self,
        state: &PublicGameState,
        before_board: &Board,
        own_board: &Board,
        prepared_own_base: &[ActiveFeature],
        prepared: PreparedOpportunityEvaluation<'_>,
        placed_wildlife: (Wildlife, HexCoord),
    ) -> Result<V3OwnFeatureSet> {
        static PROFILE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let profile = *PROFILE
            .get_or_init(|| std::env::var_os("CASCADIA_V3_PROFILE_CANDIDATE_FEATURE").is_some());
        let started = std::time::Instant::now();
        let own_base = wildlife_sibling_base_features(
            prepared_own_base,
            before_board,
            own_board,
            state,
            placed_wildlife.0,
            placed_wildlife.1,
        )?;
        let base_seconds = started.elapsed().as_secs_f64();
        let started = std::time::Instant::now();
        let (graph, assignments) = prepared.context.build_and_solve_with_matching_frontiers(
            own_board,
            prepared.habitat,
            prepared.wildlife,
            prepared.habitat_matching,
            prepared.wildlife_matching,
            Some(placed_wildlife),
        )?;
        let graph_seconds = started.elapsed().as_secs_f64();
        let started = std::time::Instant::now();
        let own_opportunities = compile_opportunity_features(
            prepared.context.state(),
            own_board,
            0,
            &graph,
            &assignments,
            true,
        )?;
        if profile {
            static PRINTED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            if PRINTED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 16 {
                eprintln!(
                    "V3_CANDIDATE_FEATURE_PROFILE {}",
                    serde_json::json!({
                        "base_seconds": base_seconds,
                        "graph_seconds": graph_seconds,
                        "compile_seconds": started.elapsed().as_secs_f64(),
                        "demands": graph.demands.len(),
                        "edges": graph.edges.len(),
                        "assignments": assignments.len(),
                        "features": own_opportunities.len(),
                    })
                );
            }
        }
        Ok(V3OwnFeatureSet {
            own_base,
            own_opportunities,
            phase_bucket: ((8 * own_board.tile_count().saturating_sub(3).min(20)) / 20).min(7)
                as u8,
        })
    }

    fn validate_opponents(&self, state: &PublicGameState) -> Result<()> {
        if state.boards().len() != 4 {
            return Err(V3Error::InvalidFeature(
                "V3 afterstate context received a non-four-player state".to_owned(),
            ));
        }
        for relative in 1..4 {
            let absolute = (self.focal_seat + relative) % 4;
            if state.boards()[absolute].canonical_hash().as_bytes()
                != &self.opponent_hashes[relative - 1]
            {
                return Err(V3Error::InvalidFeature(
                    "cached V3 opponent board changed across a focal afterstate".to_owned(),
                ));
            }
        }
        Ok(())
    }
}

impl V3FeatureSet {
    pub fn validate(&self) -> Result<()> {
        for (name, rows, width) in [
            ("own base", &self.own_base, BASE_FEATURE_ROWS),
            ("field base", &self.field_base, BASE_FEATURE_ROWS),
            (
                "own opportunities",
                &self.own_opportunities,
                FullOpportunitiesCatalog::global().len(),
            ),
            (
                "field opportunities",
                &self.field_opportunities,
                FullOpportunitiesCatalog::global().len(),
            ),
        ] {
            if rows
                .windows(2)
                .any(|window| window[0].index >= window[1].index)
                || rows
                    .iter()
                    .any(|row| row.count == 0 || row.index as usize >= width)
            {
                return Err(V3Error::InvalidFeature(format!(
                    "{name} rows are noncanonical or out of range"
                )));
            }
        }
        if self.overflow_entities.len() != 4 || self.phase_bucket > 7 {
            return Err(V3Error::InvalidFeature(
                "feature set has invalid board or phase cardinality".to_owned(),
            ));
        }
        Ok(())
    }

    pub fn natural_hot_path(&self) -> bool {
        self.overflow_entities.iter().all(Vec::is_empty)
    }
}

pub fn encode_public_features(state: &PublicGameState, focal_seat: usize) -> Result<V3FeatureSet> {
    if state.boards().len() != 4 || focal_seat >= 4 {
        return Err(V3Error::InvalidFeature(
            "V3 feature extraction requires four boards".to_owned(),
        ));
    }
    let mut board_encodings = Vec::with_capacity(4);
    let mut opportunity_encodings = Vec::with_capacity(4);
    for relative in 0..4 {
        let absolute = (focal_seat + relative) % 4;
        board_encodings.push(encode_board_features(&state.boards()[absolute])?);
        opportunity_encodings.push(opportunity_features(state, absolute, relative as u8)?);
    }
    let mut own_base = board_encodings[0].active.clone();
    own_base.extend(global_features(state, focal_seat)?);
    let field_base = consolidate(
        board_encodings[1..]
            .iter()
            .flat_map(|encoding| encoding.active.iter().copied())
            .collect(),
    )?;
    let field_opportunities = consolidate(
        opportunity_encodings[1..]
            .iter()
            .flat_map(|encoding| encoding.iter().copied())
            .collect(),
    )?;
    let completed = state.boards()[focal_seat]
        .tile_count()
        .saturating_sub(3)
        .min(20);
    let features = V3FeatureSet {
        own_base: consolidate(own_base)?,
        field_base,
        own_opportunities: opportunity_encodings.remove(0),
        field_opportunities,
        overflow_entities: board_encodings
            .into_iter()
            .map(|encoding| encoding.overflow)
            .collect(),
        phase_bucket: ((8 * completed) / 20).min(7) as u8,
    };
    features.validate()?;
    Ok(features)
}

pub fn transformed_overflow(
    entities: &[OverflowEntity],
    transform: D6Transform,
) -> Result<Vec<OverflowEntity>> {
    let mut transformed = Vec::with_capacity(entities.len());
    for entity in entities {
        let coord = transform
            .transform_coord(HexCoord::new(entity.q, entity.r))
            .map_err(|error| V3Error::InvalidFeature(error.to_string()))?;
        let mut directed = [0u8; 6];
        for edge in 0..6 {
            let target = transform
                .transform_edge(edge)
                .map_err(|error| V3Error::InvalidFeature(error.to_string()))?;
            directed[target] = entity.directed_edge_terrains[edge];
        }
        transformed.push(OverflowEntity {
            q: coord.q,
            r: coord.r,
            directed_edge_terrains: directed,
            allowed_wildlife_bits: entity.allowed_wildlife_bits,
            placed_wildlife: entity.placed_wildlife,
            keystone: entity.keystone,
        });
    }
    transformed.sort_unstable_by_key(|entity| (entity.q, entity.r));
    Ok(transformed)
}

fn transform_core_row(index: usize, transform: D6Transform) -> Result<usize> {
    let transform_cell = |cell: usize| -> Result<usize> {
        let coord = crate::hot_coord(cell).ok_or_else(|| {
            V3Error::InvalidFeature(format!("radius-7 feature references missing cell {cell}"))
        })?;
        let transformed = transform
            .transform_coord(coord)
            .map_err(|error| V3Error::InvalidFeature(error.to_string()))?;
        crate::hot_index(transformed).ok_or_else(|| {
            V3Error::InvalidFeature(format!(
                "D6 transform left radius-7 disk at {transformed:?}"
            ))
        })
    };
    if index < TERRAIN_EDGE_BASE {
        return Ok(TILE_PRESENCE_BASE + transform_cell(index - TILE_PRESENCE_BASE)?);
    }
    if index < ALLOWED_WILDLIFE_BASE {
        let local = index - TERRAIN_EDGE_BASE;
        let cell = local / 30;
        let edge = (local % 30) / 5;
        let terrain = local % 5;
        let target_edge = transform
            .transform_edge(edge)
            .map_err(|error| V3Error::InvalidFeature(error.to_string()))?;
        return Ok(TERRAIN_EDGE_BASE + (transform_cell(cell)? * 6 + target_edge) * 5 + terrain);
    }
    if index < PLACED_WILDLIFE_BASE {
        let local = index - ALLOWED_WILDLIFE_BASE;
        return Ok(ALLOWED_WILDLIFE_BASE + transform_cell(local / 5)? * 5 + local % 5);
    }
    if index < KEYSTONE_BASE {
        let local = index - PLACED_WILDLIFE_BASE;
        return Ok(PLACED_WILDLIFE_BASE + transform_cell(local / 5)? * 5 + local % 5);
    }
    if index < CORE_SPATIAL_FEATURE_ROWS {
        return Ok(KEYSTONE_BASE + transform_cell(index - KEYSTONE_BASE)?);
    }
    Err(V3Error::InvalidFeature(format!(
        "row {index} is not a core spatial feature"
    )))
}

fn transform_base_rows(
    rows: &[ActiveFeature],
    overflow_groups: &[Vec<OverflowEntity>],
    transform: D6Transform,
) -> Result<Vec<ActiveFeature>> {
    let mut transformed = Vec::with_capacity(rows.len());
    for row in rows {
        let index = row.index as usize;
        if index < CORE_SPATIAL_FEATURE_ROWS {
            transformed.push(ActiveFeature {
                index: transform_core_row(index, transform)? as u32,
                count: row.count,
            });
        } else if index >= GLOBAL_BASE {
            transformed.push(*row);
        }
        // Overflow rows are rebuilt below because slot identities can change
        // when absolute coordinates are transformed and re-sorted.
    }
    for group in overflow_groups {
        transformed.extend(encode_overflow_entities(group)?);
    }
    consolidate(transformed)
}

fn transform_opportunity_location(location: u16, transform: D6Transform) -> Result<u16> {
    if location == OVERFLOW_LOCATION {
        return Ok(location);
    }
    let cell = usize::from(location);
    let coord = crate::hot_coord(cell).ok_or_else(|| {
        V3Error::InvalidFeature(format!("opportunity references missing hot cell {cell}"))
    })?;
    let transformed = transform
        .transform_coord(coord)
        .map_err(|error| V3Error::InvalidFeature(error.to_string()))?;
    let target = crate::hot_index(transformed).ok_or_else(|| {
        V3Error::InvalidFeature("D6 opportunity location left radius-7 disk".to_owned())
    })?;
    Ok(target as u16)
}

fn transform_opportunity_rows(
    rows: &[ActiveFeature],
    transform: D6Transform,
) -> Result<Vec<ActiveFeature>> {
    let catalog = FullOpportunitiesCatalog::global();
    let mut transformed = Vec::with_capacity(rows.len());
    for row in rows {
        let spec = catalog.spec(row.index).ok_or_else(|| {
            V3Error::InvalidFeature(format!("unknown opportunity row {}", row.index))
        })?;
        let spec = match spec {
            OpportunityFeatureSpec::DemandLocation {
                relative_seat,
                demand_kind,
                subject,
                location,
            } => OpportunityFeatureSpec::DemandLocation {
                relative_seat,
                demand_kind,
                subject,
                location: transform_opportunity_location(location, transform)?,
            },
            OpportunityFeatureSpec::MarketSynergy {
                relative_seat,
                demand_kind,
                subject,
                location,
                market_slot,
            } => OpportunityFeatureSpec::MarketSynergy {
                relative_seat,
                demand_kind,
                subject,
                location: transform_opportunity_location(location, transform)?,
                market_slot,
            },
            OpportunityFeatureSpec::Completion {
                relative_seat,
                demand_kind,
                subject,
                location,
                bucket,
            } => OpportunityFeatureSpec::Completion {
                relative_seat,
                demand_kind,
                subject,
                location: transform_opportunity_location(location, transform)?,
                bucket,
            },
            coordinate_free => coordinate_free,
        };
        transformed.push(ActiveFeature {
            index: catalog.id(spec)?,
            count: row.count,
        });
    }
    consolidate(transformed)
}

/// Apply one exact D6 augmentation to an already encoded state. Global and
/// market semantics remain unchanged; spatial rows, directed terrain edges,
/// opportunity locations, and exact overflow coordinates are transformed.
pub fn transform_feature_set(
    features: &V3FeatureSet,
    transform: D6Transform,
) -> Result<V3FeatureSet> {
    features.validate()?;
    let overflow_entities = features
        .overflow_entities
        .iter()
        .map(|entities| transformed_overflow(entities, transform))
        .collect::<Result<Vec<_>>>()?;
    let transformed = V3FeatureSet {
        own_base: transform_base_rows(&features.own_base, &overflow_entities[..1], transform)?,
        field_base: transform_base_rows(&features.field_base, &overflow_entities[1..], transform)?,
        own_opportunities: transform_opportunity_rows(&features.own_opportunities, transform)?,
        field_opportunities: transform_opportunity_rows(&features.field_opportunities, transform)?,
        overflow_entities,
        phase_bucket: features.phase_bucket,
    };
    transformed.validate()?;
    Ok(transformed)
}

#[cfg(test)]
mod tests {
    use cascadia_game::{GameConfig, GameSeed, GameState, Rotation, STARTER_CLUSTERS};

    use super::*;

    #[test]
    fn opportunity_catalog_is_collision_free_and_expected_size() {
        let catalog = FullOpportunitiesCatalog::global();
        assert!((OPPORTUNITY_FEATURE_MIN..=OPPORTUNITY_FEATURE_MAX).contains(&catalog.len()));
        for (index, spec) in catalog.specs.iter().copied().enumerate() {
            assert_eq!(catalog.id(spec).unwrap(), index as u32);
            assert_eq!(catalog.spec(index as u32), Some(spec));
        }
        assert_eq!(catalog.training_factor_offsets().len(), catalog.len() + 1);
        let used = catalog
            .training_factor_indices()
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        assert_eq!(used.len(), catalog.training_factor_len());
        assert_eq!(used.first(), Some(&0));
        assert_eq!(
            used.last().copied().map(|value| value as usize + 1),
            Some(catalog.training_factor_len())
        );
        for row in 0..catalog.len() as u32 {
            let factors = catalog.training_factors_for_row(row).unwrap();
            assert!((3..=7).contains(&factors.len()));
            assert!(factors.windows(2).all(|pair| pair[0] < pair[1]));
        }
    }

    #[test]
    fn initial_four_player_state_encodes_canonically() {
        let game = GameState::new(
            GameConfig::research_aaaaa(4).unwrap(),
            GameSeed::from_u64(9001),
        )
        .unwrap();
        let features = encode_public_features(&game.public_state(), game.current_player()).unwrap();
        features.validate().unwrap();
        assert!(features.natural_hot_path());
        assert!(!features.own_base.is_empty());
        assert!(!features.own_opportunities.is_empty());
    }

    #[test]
    fn radius_eight_tile_uses_exact_overflow_features() {
        let mut board = Board::from_starter(&STARTER_CLUSTERS[0]);
        let tile = STANDARD_TILES[0];
        for q in 2..=8 {
            board
                .place_tile(HexCoord::new(q, 0), tile, Rotation::ZERO)
                .unwrap();
        }
        let encoded = encode_board_features(&board).unwrap();
        assert_eq!(encoded.overflow.len(), 1);
        assert_eq!((encoded.overflow[0].q, encoded.overflow[0].r), (8, 0));
        assert!(
            encoded.active.iter().any(|row| {
                (OVERFLOW_BASE..OVERFLOW_COUNT_BASE).contains(&(row.index as usize))
            })
        );
    }

    #[test]
    fn overflow_d6_round_trip_preserves_every_semantic_channel() {
        let entity = OverflowEntity {
            q: 9,
            r: -2,
            directed_edge_terrains: [0, 1, 2, 3, 4, 0],
            allowed_wildlife_bits: 0b10101,
            placed_wildlife: Some(3),
            keystone: false,
        };
        for transform in D6Transform::ALL {
            let transformed =
                transformed_overflow(std::slice::from_ref(&entity), transform).unwrap();
            let restored = transformed_overflow(&transformed, transform.inverse()).unwrap();
            assert_eq!(restored, vec![entity.clone()]);
        }
    }

    #[test]
    fn complete_feature_set_d6_round_trip_is_exact() {
        let game = GameState::new(
            GameConfig::research_aaaaa(4).unwrap(),
            GameSeed::from_u64(9002),
        )
        .unwrap();
        let features = encode_public_features(&game.public_state(), 0).unwrap();
        for transform in D6Transform::ALL {
            let augmented = transform_feature_set(&features, transform).unwrap();
            let restored = transform_feature_set(&augmented, transform.inverse()).unwrap();
            assert_eq!(restored, features);
        }
    }
}
