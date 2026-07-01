use std::sync::OnceLock;

use cascadia_game::{D6Transform, HexCoord};
use serde::{Deserialize, Serialize};

use crate::{FullOpportunitiesCatalog, Result, V3Error};

pub const V3_FEATURE_SCHEMA_VERSION: u16 = 1;
pub const V3_FEATURE_SCHEMA_ID: &str = "cascadia-v3-radius7-nnue-features-v1";
pub const HOT_RADIUS: u8 = 7;
pub const HOT_CELL_COUNT: usize = 1 + 3 * 7 * 8;
pub const OVERFLOW_SLOT_COUNT: usize = 23;
pub const OVERFLOW_COORD_MIN: i8 = -21;
pub const OVERFLOW_COORD_MAX: i8 = 21;
pub const OVERFLOW_COORD_BINS: usize = 43;

pub const TILE_PRESENCE_BASE: usize = 0;
pub const TERRAIN_EDGE_BASE: usize = TILE_PRESENCE_BASE + HOT_CELL_COUNT;
pub const ALLOWED_WILDLIFE_BASE: usize = TERRAIN_EDGE_BASE + HOT_CELL_COUNT * 6 * 5;
pub const PLACED_WILDLIFE_BASE: usize = ALLOWED_WILDLIFE_BASE + HOT_CELL_COUNT * 5;
pub const KEYSTONE_BASE: usize = PLACED_WILDLIFE_BASE + HOT_CELL_COUNT * 5;
pub const CORE_SPATIAL_FEATURE_ROWS: usize = KEYSTONE_BASE + HOT_CELL_COUNT;

pub const OVERFLOW_SLOT_WIDTH: usize =
    1 + OVERFLOW_COORD_BINS + OVERFLOW_COORD_BINS + 6 * 5 + 5 + 5 + 1;
pub const OVERFLOW_BASE: usize = CORE_SPATIAL_FEATURE_ROWS;
pub const OVERFLOW_COUNT_BASE: usize = OVERFLOW_BASE + OVERFLOW_SLOT_COUNT * OVERFLOW_SLOT_WIDTH;
pub const GLOBAL_BASE: usize = OVERFLOW_COUNT_BASE + OVERFLOW_SLOT_COUNT + 1;
pub const GLOBAL_FEATURE_ROWS: usize = 1_703;
pub const BASE_FEATURE_ROWS: usize = GLOBAL_BASE + GLOBAL_FEATURE_ROWS;

pub const OPPORTUNITY_FEATURE_MIN: usize = 80_000;
pub const OPPORTUNITY_FEATURE_MAX: usize = 95_000;

const _: () = {
    assert!(HOT_CELL_COUNT == 169);
    assert!(CORE_SPATIAL_FEATURE_ROWS == 7_098);
    assert!(OVERFLOW_SLOT_WIDTH == 128);
    assert!(BASE_FEATURE_ROWS == 11_769);
};

fn build_hot_coords() -> Vec<HexCoord> {
    let mut coordinates = Vec::with_capacity(HOT_CELL_COUNT);
    for q in -(HOT_RADIUS as i8)..=HOT_RADIUS as i8 {
        for r in -(HOT_RADIUS as i8)..=HOT_RADIUS as i8 {
            let coord = HexCoord::new(q, r);
            if coord.radius() <= HOT_RADIUS {
                coordinates.push(coord);
            }
        }
    }
    assert_eq!(coordinates.len(), HOT_CELL_COUNT);
    coordinates
}

pub fn hot_coords() -> &'static [HexCoord] {
    static COORDS: OnceLock<Vec<HexCoord>> = OnceLock::new();
    COORDS.get_or_init(build_hot_coords)
}

pub fn hot_coord(index: usize) -> Option<HexCoord> {
    hot_coords().get(index).copied()
}

pub fn hot_index(coord: HexCoord) -> Option<usize> {
    if coord.radius() > HOT_RADIUS {
        return None;
    }
    hot_coords()
        .binary_search_by_key(&(coord.q, coord.r), |value| (value.q, value.r))
        .ok()
}

fn d6_permutations() -> Result<Vec<Vec<u16>>> {
    D6Transform::ALL
        .into_iter()
        .map(|transform| {
            hot_coords()
                .iter()
                .map(|coord| {
                    let transformed = transform.transform_coord(*coord).map_err(|error| {
                        V3Error::InvalidFeature(format!(
                            "D6 transform {} failed for {coord:?}: {error}",
                            transform.id()
                        ))
                    })?;
                    let index = hot_index(transformed).ok_or_else(|| {
                        V3Error::InvalidFeature(format!(
                            "D6 transform {} left the radius-7 disk at {transformed:?}",
                            transform.id()
                        ))
                    })?;
                    u16::try_from(index).map_err(|_| {
                        V3Error::InvalidFeature("radius-7 index exceeds u16".to_owned())
                    })
                })
                .collect()
        })
        .collect()
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct V3FeatureSchemaManifest {
    pub schema_version: u16,
    pub schema_id: String,
    pub hot_radius: u8,
    pub hot_cell_count: u16,
    pub coordinate_order: Vec<HexCoord>,
    pub d6_transform_ids: Vec<u8>,
    pub d6_hot_index_permutations: Vec<Vec<u16>>,
    pub overflow_slot_count: u8,
    pub overflow_coordinate_min: i8,
    pub overflow_coordinate_max: i8,
    pub core_spatial_feature_rows: u32,
    pub global_feature_rows: u32,
    pub base_feature_rows: u32,
    pub opportunity_feature_rows: u32,
    pub opportunity_catalog_blake3: String,
    pub opportunity_training_factor_rows: u32,
    pub opportunity_training_factor_blake3: String,
    pub opportunity_training_factor_offsets: Vec<u32>,
    pub opportunity_training_factor_indices: Vec<u32>,
    pub canonical_blake3: String,
}

impl V3FeatureSchemaManifest {
    pub fn build() -> Result<Self> {
        let catalog = FullOpportunitiesCatalog::global();
        let mut manifest = Self {
            schema_version: V3_FEATURE_SCHEMA_VERSION,
            schema_id: V3_FEATURE_SCHEMA_ID.to_owned(),
            hot_radius: HOT_RADIUS,
            hot_cell_count: HOT_CELL_COUNT as u16,
            coordinate_order: hot_coords().to_vec(),
            d6_transform_ids: D6Transform::ALL.into_iter().map(D6Transform::id).collect(),
            d6_hot_index_permutations: d6_permutations()?,
            overflow_slot_count: OVERFLOW_SLOT_COUNT as u8,
            overflow_coordinate_min: OVERFLOW_COORD_MIN,
            overflow_coordinate_max: OVERFLOW_COORD_MAX,
            core_spatial_feature_rows: CORE_SPATIAL_FEATURE_ROWS as u32,
            global_feature_rows: GLOBAL_FEATURE_ROWS as u32,
            base_feature_rows: BASE_FEATURE_ROWS as u32,
            opportunity_feature_rows: catalog.len() as u32,
            opportunity_catalog_blake3: catalog.checksum().to_owned(),
            opportunity_training_factor_rows: catalog.training_factor_len() as u32,
            opportunity_training_factor_blake3: catalog.training_factor_checksum().to_owned(),
            opportunity_training_factor_offsets: catalog.training_factor_offsets().to_vec(),
            opportunity_training_factor_indices: catalog.training_factor_indices().to_vec(),
            canonical_blake3: String::new(),
        };
        let bytes = serde_json::to_vec(&manifest)?;
        manifest.canonical_blake3 = blake3::hash(&bytes).to_hex().to_string();
        Ok(manifest)
    }

    pub fn validate(&self) -> Result<()> {
        let expected = Self::build()?;
        if self != &expected {
            return Err(V3Error::InvalidFeature(
                "feature schema manifest differs from the canonical V3 schema".to_owned(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;

    #[test]
    fn radius_seven_has_169_stably_indexed_cells() {
        assert_eq!(hot_coords().len(), 169);
        assert_eq!(
            hot_coords().iter().copied().collect::<BTreeSet<_>>().len(),
            169
        );
        for (index, coord) in hot_coords().iter().copied().enumerate() {
            assert_eq!(hot_index(coord), Some(index));
            assert_eq!(hot_coord(index), Some(coord));
        }
    }

    #[test]
    fn every_d6_transform_is_a_radius_seven_permutation() {
        let manifest = V3FeatureSchemaManifest::build().unwrap();
        for permutation in manifest.d6_hot_index_permutations {
            assert_eq!(permutation.len(), HOT_CELL_COUNT);
            assert_eq!(
                permutation.iter().copied().collect::<BTreeSet<_>>().len(),
                169
            );
        }
    }

    #[test]
    fn declared_feature_dimensions_are_exact() {
        assert_eq!(CORE_SPATIAL_FEATURE_ROWS, 7_098);
        assert_eq!(BASE_FEATURE_ROWS, 11_769);
    }
}
