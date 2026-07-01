use std::collections::HashMap;

use serde::{Deserialize, Deserializer, Serialize, Serializer, de};
use thiserror::Error;

use crate::{GameState, HexCoord, HexDirection, MarketPrelude, Rotation, RuleError, Tile};

pub const D6_CONTRACT_SCHEMA_VERSION: u16 = 1;

/// One element `T(k, f) = R^k S^f` of the hexagonal dihedral group.
///
/// `R(q, r) = (q + r, -q)` and `S(q, r) = (q + r, -r)`. Stable IDs are
/// rotations `0..=5`, followed by reflected transforms `6..=11`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct D6Transform {
    rotation: u8,
    reflected: bool,
}

impl Serialize for D6Transform {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u8(self.id())
    }
}

impl<'de> Deserialize<'de> for D6Transform {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let id = u8::deserialize(deserializer)?;
        Self::from_id(id).ok_or_else(|| de::Error::custom("D6 transform ID must be in [0, 11]"))
    }
}

impl D6Transform {
    pub const IDENTITY: Self = Self::new_unchecked(0, false);
    pub const ALL: [Self; 12] = [
        Self::new_unchecked(0, false),
        Self::new_unchecked(1, false),
        Self::new_unchecked(2, false),
        Self::new_unchecked(3, false),
        Self::new_unchecked(4, false),
        Self::new_unchecked(5, false),
        Self::new_unchecked(0, true),
        Self::new_unchecked(1, true),
        Self::new_unchecked(2, true),
        Self::new_unchecked(3, true),
        Self::new_unchecked(4, true),
        Self::new_unchecked(5, true),
    ];
    pub const INVERSE_TABLE: [u8; 12] = build_inverse_table();
    pub const COMPOSITION_TABLE: [[u8; 12]; 12] = build_composition_table();

    pub const fn new(rotation: u8, reflected: bool) -> Option<Self> {
        if rotation < 6 {
            Some(Self {
                rotation,
                reflected,
            })
        } else {
            None
        }
    }

    const fn new_unchecked(rotation: u8, reflected: bool) -> Self {
        Self {
            rotation,
            reflected,
        }
    }

    pub const fn from_id(id: u8) -> Option<Self> {
        if id < 12 {
            Some(Self::new_unchecked(id % 6, id >= 6))
        } else {
            None
        }
    }

    pub const fn id(self) -> u8 {
        self.rotation + if self.reflected { 6 } else { 0 }
    }

    pub const fn rotation_steps(self) -> u8 {
        self.rotation
    }

    pub const fn is_reflected(self) -> bool {
        self.reflected
    }

    pub const fn inverse(self) -> Self {
        match Self::from_id(Self::INVERSE_TABLE[self.id() as usize]) {
            Some(inverse) => inverse,
            None => unreachable_const(),
        }
    }

    /// Returns `self` after `right`: `result(x) = self(right(x))`.
    pub const fn compose(self, right: Self) -> Self {
        match Self::from_id(Self::COMPOSITION_TABLE[self.id() as usize][right.id() as usize]) {
            Some(composed) => composed,
            None => unreachable_const(),
        }
    }

    pub fn transform_coord(self, coord: HexCoord) -> Result<HexCoord, D6Error> {
        let (mut q, mut r) = (i16::from(coord.q), i16::from(coord.r));
        if self.reflected {
            (q, r) = (q + r, -r);
        }
        for _ in 0..self.rotation {
            (q, r) = (q + r, -q);
        }
        let q = i8::try_from(q).map_err(|_| D6Error::CoordinateOverflow {
            transform: self,
            source_coord: coord,
        })?;
        let r = i8::try_from(r).map_err(|_| D6Error::CoordinateOverflow {
            transform: self,
            source_coord: coord,
        })?;
        Ok(HexCoord::new(q, r))
    }

    pub const fn transform_direction(self, direction: HexDirection) -> HexDirection {
        let edge = direction as i8;
        let transformed = if self.reflected {
            modulo_six(self.rotation as i8 - edge)
        } else {
            modulo_six(edge + self.rotation as i8)
        };
        HexDirection::ALL[transformed as usize]
    }

    pub fn transform_edge(self, edge: usize) -> Result<usize, D6Error> {
        let direction = HexDirection::from_index(edge).ok_or(D6Error::InvalidEdge(edge))?;
        Ok(self.transform_direction(direction).index())
    }

    pub const fn transform_tile_rotation(self, tile: Tile, rotation: Rotation) -> Rotation {
        if tile.terrain_b.is_none() {
            return Rotation::ZERO;
        }
        let transformed = if self.reflected {
            modulo_six(self.rotation as i8 - rotation.get() as i8 - 2)
        } else {
            modulo_six(rotation.get() as i8 + self.rotation as i8)
        };
        Rotation::ALL[transformed as usize]
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum D6Error {
    #[error("edge index {0} is outside the canonical six-edge domain")]
    InvalidEdge(usize),
    #[error("{transform:?} overflows i8 while transforming {source_coord:?}")]
    CoordinateOverflow {
        transform: D6Transform,
        source_coord: HexCoord,
    },
    #[error(
        "{transform:?} maps board coordinate {source_coord:?} to unsupported coordinate {transformed:?}"
    )]
    BoardCoordinateOutOfBounds {
        transform: D6Transform,
        source_coord: HexCoord,
        transformed: HexCoord,
    },
    #[error("multiple occupied coordinates transformed to {0:?}")]
    BoardCoordinateCollision(HexCoord),
    #[error("transformed frontier differs from the exact image under {0:?}")]
    FrontierMismatch(D6Transform),
    #[error(
        "source and transformed legal sets have different sizes: {source_len} versus {transformed_len}"
    )]
    LegalActionCardinality {
        source_len: usize,
        transformed_len: usize,
    },
    #[error("source legal set contains duplicate action value at row {0}")]
    DuplicateSourceAction(usize),
    #[error("transformed legal set contains duplicate action value at row {0}")]
    DuplicateTransformedAction(usize),
    #[error("source legal action row {0} has no transformed action value")]
    MissingTransformedAction(usize),
    #[error("legal-action row map is not bijective")]
    NonBijectiveActionMap,
    #[error("policy vector has length {actual}; expected {expected}")]
    PolicyLength { expected: usize, actual: usize },
    #[error(transparent)]
    Rule(#[from] RuleError),
    #[error("transformed state invariant failed: {0}")]
    Invariant(&'static str),
}

/// Exact source-row to transformed-row and transformed-row to source-row maps.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LegalActionPermutation {
    transform: D6Transform,
    forward: Vec<usize>,
    inverse: Vec<usize>,
}

impl LegalActionPermutation {
    pub fn new(
        state: &GameState,
        prelude: &MarketPrelude,
        transform: D6Transform,
    ) -> Result<Self, D6Error> {
        let source_actions = state.legal_turn_actions(prelude)?;
        let transformed_state = state.transformed(transform)?;
        let transformed_actions = transformed_state.legal_turn_actions(prelude)?;
        if source_actions.len() != transformed_actions.len() {
            return Err(D6Error::LegalActionCardinality {
                source_len: source_actions.len(),
                transformed_len: transformed_actions.len(),
            });
        }

        let mut source_uniqueness = HashMap::with_capacity(source_actions.len());
        for (row, action) in source_actions.iter().enumerate() {
            if source_uniqueness.insert(action, row).is_some() {
                return Err(D6Error::DuplicateSourceAction(row));
            }
        }

        let mut transformed_rows = HashMap::with_capacity(transformed_actions.len());
        for (row, action) in transformed_actions.iter().enumerate() {
            if transformed_rows.insert(action, row).is_some() {
                return Err(D6Error::DuplicateTransformedAction(row));
            }
        }

        let mut forward = Vec::with_capacity(source_actions.len());
        let mut inverse = vec![usize::MAX; source_actions.len()];
        for (source_row, action) in source_actions.iter().enumerate() {
            let transformed_action = state.transform_turn_action(action, transform)?;
            let transformed_row = transformed_rows
                .get(&transformed_action)
                .copied()
                .ok_or(D6Error::MissingTransformedAction(source_row))?;
            if inverse[transformed_row] != usize::MAX {
                return Err(D6Error::NonBijectiveActionMap);
            }
            forward.push(transformed_row);
            inverse[transformed_row] = source_row;
        }
        if inverse.contains(&usize::MAX) {
            return Err(D6Error::NonBijectiveActionMap);
        }

        Ok(Self {
            transform,
            forward,
            inverse,
        })
    }

    pub fn transform(&self) -> D6Transform {
        self.transform
    }

    pub fn len(&self) -> usize {
        self.forward.len()
    }

    pub fn is_empty(&self) -> bool {
        self.forward.is_empty()
    }

    pub fn forward_rows(&self) -> &[usize] {
        &self.forward
    }

    pub fn inverse_rows(&self) -> &[usize] {
        &self.inverse
    }

    /// Places each source value at its transformed legal-action row.
    pub fn permute_forward<T: Clone>(&self, source: &[T]) -> Result<Vec<T>, D6Error> {
        self.validate_policy_len(source.len())?;
        Ok(self
            .inverse
            .iter()
            .map(|source_row| source[*source_row].clone())
            .collect())
    }

    /// Restores source-row order from values indexed by transformed rows.
    pub fn permute_inverse<T: Clone>(&self, transformed: &[T]) -> Result<Vec<T>, D6Error> {
        self.validate_policy_len(transformed.len())?;
        Ok(self
            .forward
            .iter()
            .map(|transformed_row| transformed[*transformed_row].clone())
            .collect())
    }

    pub fn validate(&self) -> Result<(), D6Error> {
        if self.forward.len() != self.inverse.len() {
            return Err(D6Error::NonBijectiveActionMap);
        }
        for (source, transformed) in self.forward.iter().copied().enumerate() {
            if transformed >= self.inverse.len() || self.inverse[transformed] != source {
                return Err(D6Error::NonBijectiveActionMap);
            }
        }
        Ok(())
    }

    fn validate_policy_len(&self, actual: usize) -> Result<(), D6Error> {
        if actual == self.len() {
            Ok(())
        } else {
            Err(D6Error::PolicyLength {
                expected: self.len(),
                actual,
            })
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct D6TransformMetadata {
    pub id: u8,
    pub rotation_steps: u8,
    pub reflected: bool,
    pub name: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct D6ContractMetadata {
    pub schema_version: u16,
    pub contract_id: String,
    pub edge_order: [String; 6],
    pub coordinate_matrices: [[[i8; 2]; 2]; 12],
    pub direction_tables: [[u8; 6]; 12],
    pub dual_tile_rotation_tables: [[u8; 6]; 12],
    pub single_tile_rotation_tables: [[u8; 6]; 12],
    pub inverse_table: [u8; 12],
    pub composition_table: [[u8; 12]; 12],
    pub transforms: [D6TransformMetadata; 12],
    pub scientific_blake3: String,
}

#[derive(Serialize)]
struct D6ScientificPayload<'a> {
    schema_version: u16,
    contract_id: &'a str,
    edge_order: &'a [String; 6],
    coordinate_matrices: &'a [[[i8; 2]; 2]; 12],
    direction_tables: &'a [[u8; 6]; 12],
    dual_tile_rotation_tables: &'a [[u8; 6]; 12],
    single_tile_rotation_tables: &'a [[u8; 6]; 12],
    inverse_table: &'a [u8; 12],
    composition_table: &'a [[u8; 12]; 12],
    transforms: &'a [D6TransformMetadata; 12],
}

pub fn d6_contract_metadata() -> D6ContractMetadata {
    let contract_id = "cascadia-game-exact-d6-v1".to_owned();
    let edge_order = HexDirection::NAMES.map(str::to_owned);
    let coordinate_matrices = std::array::from_fn(|index| {
        let transform = D6Transform::ALL[index];
        let q_basis = transform
            .transform_coord(HexCoord::new(1, 0))
            .expect("basis transform fits i8");
        let r_basis = transform
            .transform_coord(HexCoord::new(0, 1))
            .expect("basis transform fits i8");
        [[q_basis.q, r_basis.q], [q_basis.r, r_basis.r]]
    });
    let direction_tables = std::array::from_fn(|transform_index| {
        std::array::from_fn(|edge| {
            D6Transform::ALL[transform_index].transform_direction(HexDirection::ALL[edge]) as u8
        })
    });
    let dual_tile = Tile::dual(u8::MAX, crate::Terrain::Mountain, crate::Terrain::Forest, 0);
    let single_tile = Tile::keystone(u8::MAX, crate::Terrain::Mountain, crate::Wildlife::Bear);
    let dual_tile_rotation_tables = std::array::from_fn(|transform_index| {
        std::array::from_fn(|rotation| {
            D6Transform::ALL[transform_index]
                .transform_tile_rotation(dual_tile, Rotation::ALL[rotation])
                .get()
        })
    });
    let single_tile_rotation_tables = std::array::from_fn(|transform_index| {
        std::array::from_fn(|rotation| {
            D6Transform::ALL[transform_index]
                .transform_tile_rotation(single_tile, Rotation::ALL[rotation])
                .get()
        })
    });
    let transforms = std::array::from_fn(|index| {
        let transform = D6Transform::ALL[index];
        D6TransformMetadata {
            id: transform.id(),
            rotation_steps: transform.rotation_steps(),
            reflected: transform.is_reflected(),
            name: if transform.is_reflected() {
                format!("R{}S", transform.rotation_steps())
            } else {
                format!("R{}", transform.rotation_steps())
            },
        }
    });

    let mut metadata = D6ContractMetadata {
        schema_version: D6_CONTRACT_SCHEMA_VERSION,
        contract_id,
        edge_order,
        coordinate_matrices,
        direction_tables,
        dual_tile_rotation_tables,
        single_tile_rotation_tables,
        inverse_table: D6Transform::INVERSE_TABLE,
        composition_table: D6Transform::COMPOSITION_TABLE,
        transforms,
        scientific_blake3: String::new(),
    };
    let payload = D6ScientificPayload {
        schema_version: metadata.schema_version,
        contract_id: &metadata.contract_id,
        edge_order: &metadata.edge_order,
        coordinate_matrices: &metadata.coordinate_matrices,
        direction_tables: &metadata.direction_tables,
        dual_tile_rotation_tables: &metadata.dual_tile_rotation_tables,
        single_tile_rotation_tables: &metadata.single_tile_rotation_tables,
        inverse_table: &metadata.inverse_table,
        composition_table: &metadata.composition_table,
        transforms: &metadata.transforms,
    };
    let bytes = postcard::to_allocvec(&payload).expect("D6 metadata serialization cannot fail");
    metadata.scientific_blake3 = blake3::hash(&bytes).to_hex().to_string();
    metadata
}

const fn modulo_six(value: i8) -> u8 {
    value.rem_euclid(6) as u8
}

const fn compose_id(left: u8, right: u8) -> u8 {
    let left_rotation = left % 6;
    let right_rotation = right % 6;
    let left_reflected = left >= 6;
    let right_reflected = right >= 6;
    let rotation = if left_reflected {
        modulo_six(left_rotation as i8 - right_rotation as i8)
    } else {
        modulo_six(left_rotation as i8 + right_rotation as i8)
    };
    rotation
        + if left_reflected != right_reflected {
            6
        } else {
            0
        }
}

const fn build_composition_table() -> [[u8; 12]; 12] {
    let mut table = [[0; 12]; 12];
    let mut left = 0;
    while left < 12 {
        let mut right = 0;
        while right < 12 {
            table[left][right] = compose_id(left as u8, right as u8);
            right += 1;
        }
        left += 1;
    }
    table
}

const fn build_inverse_table() -> [u8; 12] {
    let mut table = [0; 12];
    let mut id = 0;
    while id < 12 {
        table[id] = if id >= 6 {
            id as u8
        } else {
            modulo_six(-(id as i8))
        };
        id += 1;
    }
    table
}

const fn unreachable_const<T>() -> T {
    panic!("invalid constant D6 table")
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use crate::{
        Board, DraftChoice, GameConfig, GameSeed, MarketSlot, STANDARD_TILES, STARTER_CLUSTERS,
        ScoringCards, ScoringVariant, Terrain, TurnAction, Wildlife, WildlifeWipe, score_board,
        score_game,
    };

    use super::*;

    fn dual_tile(id: u8) -> Tile {
        Tile::dual(
            id,
            Terrain::Forest,
            Terrain::River,
            (1 << Wildlife::Bear as u8)
                | (1 << Wildlife::Elk as u8)
                | (1 << Wildlife::Salmon as u8)
                | (1 << Wildlife::Hawk as u8)
                | (1 << Wildlife::Fox as u8),
        )
    }

    fn fixture_board() -> Board {
        let mut board = Board::empty();
        let keystone = Tile::keystone(240, Terrain::Mountain, Wildlife::Bear);
        board
            .place_tile(HexCoord::ORIGIN, keystone, Rotation::FIVE)
            .unwrap();
        board
            .place_wildlife(HexCoord::ORIGIN, Wildlife::Bear)
            .unwrap();

        let placements = [
            (HexCoord::new(1, 0), Rotation::ONE, Wildlife::Elk),
            (HexCoord::new(1, -1), Rotation::TWO, Wildlife::Salmon),
            (HexCoord::new(0, -1), Rotation::THREE, Wildlife::Hawk),
            (HexCoord::new(-1, 0), Rotation::FOUR, Wildlife::Fox),
            (HexCoord::new(-1, 1), Rotation::FIVE, Wildlife::Bear),
            (HexCoord::new(0, 1), Rotation::ZERO, Wildlife::Salmon),
        ];
        for (index, (coord, rotation, wildlife)) in placements.into_iter().enumerate() {
            board
                .place_tile(coord, dual_tile(241 + index as u8), rotation)
                .unwrap();
            board.place_wildlife(coord, wildlife).unwrap();
        }
        board
    }

    fn game(seed: u64) -> GameState {
        GameState::new(
            GameConfig::research_aaaaa(4).unwrap(),
            GameSeed::from_u64(seed),
        )
        .unwrap()
    }

    fn generated_state(seed: u64, turns: usize) -> GameState {
        let mut state = game(seed);
        for turn in 0..turns {
            let actions = state.legal_turn_actions(&MarketPrelude::default()).unwrap();
            let action = actions[(turn * 7_919 + seed as usize) % actions.len()].clone();
            state.apply(&action).unwrap();
        }
        state
    }

    fn all_scoring_cards() -> Vec<ScoringCards> {
        let variants = [
            ScoringVariant::A,
            ScoringVariant::B,
            ScoringVariant::C,
            ScoringVariant::D,
        ];
        let mut cards = Vec::with_capacity(4usize.pow(5));
        for bear in variants {
            for elk in variants {
                for salmon in variants {
                    for hawk in variants {
                        for fox in variants {
                            cards.push(ScoringCards {
                                bear,
                                elk,
                                salmon,
                                hawk,
                                fox,
                            });
                        }
                    }
                }
            }
        }
        cards
    }

    fn state_with_paid_prelude() -> GameState {
        for seed in 0..256 {
            let mut state = GameState::new(
                GameConfig::research_aaaaa(2).unwrap(),
                GameSeed::from_u64(10_000 + seed),
            )
            .unwrap();
            let actions = state.legal_turn_actions(&MarketPrelude::default()).unwrap();
            let Some(token_action) = actions.into_iter().find(|action| {
                state
                    .preview_active_board(action)
                    .is_ok_and(|board| board.nature_tokens() > 0)
            }) else {
                continue;
            };
            state.apply(&token_action).unwrap();
            let reply = state
                .legal_turn_actions(&MarketPrelude::default())
                .unwrap()
                .into_iter()
                .next()
                .unwrap();
            state.apply(&reply).unwrap();
            if state.boards()[state.current_player()].nature_tokens() > 0 {
                return state;
            }
        }
        panic!("expected a deterministic seed with an active nature token");
    }

    fn state_with_free_three_of_a_kind() -> GameState {
        (0..2_048)
            .map(|seed| game(20_000 + seed))
            .find(|state| state.market().three_of_a_kind().is_some())
            .expect("expected a deterministic initial three-of-a-kind market")
    }

    #[test]
    fn all_elements_have_unique_stable_ids_and_metadata() {
        let ids: HashSet<_> = D6Transform::ALL.into_iter().map(D6Transform::id).collect();
        assert_eq!(ids.len(), 12);
        assert_eq!(D6Transform::IDENTITY.id(), 0);
        for (id, transform) in D6Transform::ALL.into_iter().enumerate() {
            assert_eq!(transform.id(), id as u8);
            assert_eq!(D6Transform::from_id(id as u8), Some(transform));
        }
        assert_eq!(D6Transform::from_id(12), None);
        assert_eq!(D6Transform::new(6, false), None);
        assert_eq!(
            serde_json::to_value(D6Transform::new(2, true).unwrap()).unwrap(),
            8
        );
        assert!(serde_json::from_str::<D6Transform>("12").is_err());

        let metadata = d6_contract_metadata();
        assert_eq!(metadata.schema_version, D6_CONTRACT_SCHEMA_VERSION);
        assert_eq!(metadata.edge_order, ["E", "NE", "NW", "W", "SW", "SE"]);
        assert_eq!(metadata.transforms.len(), 12);
        assert_eq!(metadata.scientific_blake3.len(), 64);
        assert_eq!(
            metadata.scientific_blake3,
            "db6ac2f9f6ebe2daaa2db603c6c16183512b5d989aed6979e1991e167737633f"
        );
        assert_eq!(
            metadata.scientific_blake3,
            d6_contract_metadata().scientific_blake3
        );
        let json = serde_json::to_value(&metadata).unwrap();
        assert_eq!(json["contract_id"], "cascadia-game-exact-d6-v1");
        assert_eq!(json["composition_table"].as_array().unwrap().len(), 12);
    }

    #[test]
    fn group_tables_match_identity_inverse_composition_and_associativity() {
        let probe = HexCoord::new(4, -7);
        for left in D6Transform::ALL {
            assert_eq!(left.compose(D6Transform::IDENTITY), left);
            assert_eq!(D6Transform::IDENTITY.compose(left), left);
            assert_eq!(left.compose(left.inverse()), D6Transform::IDENTITY);
            assert_eq!(left.inverse().compose(left), D6Transform::IDENTITY);
            assert_eq!(
                D6Transform::INVERSE_TABLE[left.id() as usize],
                left.inverse().id()
            );

            for right in D6Transform::ALL {
                let composed = left.compose(right);
                assert_eq!(
                    D6Transform::COMPOSITION_TABLE[left.id() as usize][right.id() as usize],
                    composed.id()
                );
                assert_eq!(
                    composed.transform_coord(probe).unwrap(),
                    left.transform_coord(right.transform_coord(probe).unwrap())
                        .unwrap()
                );
                for third in D6Transform::ALL {
                    assert_eq!(
                        left.compose(right).compose(third),
                        left.compose(right.compose(third))
                    );
                }
            }
        }
    }

    #[test]
    fn coordinates_round_trip_and_preserve_hex_geometry() {
        let coords: Vec<_> = (-8..=8)
            .flat_map(|q| (-8..=8).map(move |r| HexCoord::new(q, r)))
            .filter(|coord| coord.radius() <= 8)
            .collect();
        for transform in D6Transform::ALL {
            for &left in &coords {
                let transformed = transform.transform_coord(left).unwrap();
                assert_eq!(
                    transform.inverse().transform_coord(transformed).unwrap(),
                    left
                );
                assert_eq!(transformed.radius(), left.radius());
                for direction in HexDirection::ALL {
                    let neighbor = left.neighbor_in(direction);
                    assert_eq!(
                        transform.transform_coord(neighbor).unwrap(),
                        transformed.neighbor_in(transform.transform_direction(direction))
                    );
                }
                for &right in &coords {
                    assert_eq!(
                        transformed.distance(transform.transform_coord(right).unwrap()),
                        left.distance(right)
                    );
                }
            }
        }
    }

    #[test]
    fn directions_edges_and_opposites_transform_exactly() {
        for transform in D6Transform::ALL {
            assert_eq!(transform.transform_edge(6), Err(D6Error::InvalidEdge(6)));
            for direction in HexDirection::ALL {
                let transformed = transform.transform_direction(direction);
                assert_eq!(
                    transform.transform_edge(direction.index()).unwrap(),
                    transformed.index()
                );
                assert_eq!(
                    transform.transform_direction(direction.opposite()),
                    transformed.opposite()
                );
            }
        }
    }

    #[test]
    fn tile_edge_covariance_holds_for_every_rotation_edge_and_transform() {
        let tiles = STANDARD_TILES
            .into_iter()
            .chain(
                STARTER_CLUSTERS
                    .into_iter()
                    .flatten()
                    .map(|placed| placed.tile),
            )
            .chain([dual_tile(230)]);
        for tile in tiles {
            for transform in D6Transform::ALL {
                for rotation in Rotation::ALL {
                    let transformed_rotation = transform.transform_tile_rotation(tile, rotation);
                    if tile.terrain_b.is_none() {
                        assert_eq!(transformed_rotation, Rotation::ZERO);
                    }
                    for edge in 0..6 {
                        let transformed_edge = transform.transform_edge(edge).unwrap();
                        assert_eq!(
                            tile.terrain_on_edge(transformed_rotation, transformed_edge),
                            tile.terrain_on_edge(rotation, edge)
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn board_roundtrip_frontier_scoring_and_habitats_are_invariant() {
        let board = fixture_board();
        assert_eq!(
            board.tile_at(HexCoord::ORIGIN).unwrap().rotation,
            Rotation::ZERO,
            "single-terrain rotations canonicalize at insertion"
        );
        let cards = all_scoring_cards();

        for transform in D6Transform::ALL {
            let transformed = board.transformed(transform).unwrap();
            assert_eq!(transformed.nature_tokens(), board.nature_tokens());
            assert_eq!(transformed.tile_count(), board.tile_count());
            assert_eq!(transformed.transformed(transform.inverse()).unwrap(), board);

            let expected_frontier: HashSet<_> = board
                .frontier()
                .into_iter()
                .map(|coord| transform.transform_coord(coord).unwrap())
                .collect();
            assert_eq!(
                transformed.frontier().into_iter().collect::<HashSet<_>>(),
                expected_frontier
            );
            for terrain in Terrain::ALL {
                assert_eq!(
                    transformed.largest_habitat(terrain),
                    board.largest_habitat(terrain)
                );
            }
            assert_eq!(
                transformed.habitat_analysis().matching_edges(),
                board.habitat_analysis().matching_edges()
            );
            for &scoring_cards in &cards {
                assert_eq!(
                    score_board(&transformed, scoring_cards),
                    score_board(&board, scoring_cards)
                );
            }
        }
    }

    #[test]
    fn board_transform_reports_finite_grid_overflow_without_clipping() {
        let mut board = Board::empty();
        board
            .place_tile(HexCoord::new(24, 24), dual_tile(229), Rotation::ZERO)
            .unwrap();
        assert!(matches!(
            board.transformed(D6Transform::new(1, false).unwrap()),
            Err(D6Error::BoardCoordinateOutOfBounds { .. })
        ));
    }

    #[test]
    fn complete_state_roundtrip_preserves_hidden_order_rules_and_scores() {
        let state = generated_state(31_337, 7);
        let original_scores = score_game(&state);
        for transform in D6Transform::ALL {
            let transformed = state.transformed(transform).unwrap();
            assert_eq!(transformed.config(), state.config());
            assert_eq!(transformed.seed(), state.seed());
            assert_eq!(transformed.market(), state.market());
            assert_eq!(transformed.current_player(), state.current_player());
            assert_eq!(transformed.completed_turns(), state.completed_turns());
            assert_eq!(transformed.public_supply(), state.public_supply());
            assert_eq!(score_game(&transformed), original_scores);
            assert_eq!(transformed.transformed(transform.inverse()).unwrap(), state);
            assert_eq!(
                state
                    .public_state()
                    .transformed(transform)
                    .unwrap()
                    .transformed(transform.inverse())
                    .unwrap(),
                state.public_state()
            );
        }
    }

    #[test]
    fn legal_action_sets_are_exact_bijections_on_initial_and_generated_states() {
        let states = [game(41), generated_state(42, 5), generated_state(43, 11)];
        for state in &states {
            for transform in D6Transform::ALL {
                let permutation =
                    LegalActionPermutation::new(state, &MarketPrelude::default(), transform)
                        .unwrap();
                permutation.validate().unwrap();
                assert!(!permutation.is_empty());
                if transform == D6Transform::IDENTITY {
                    assert_eq!(
                        permutation.forward_rows(),
                        (0..permutation.len()).collect::<Vec<_>>()
                    );
                }
                let forward: HashSet<_> = permutation.forward_rows().iter().copied().collect();
                let inverse: HashSet<_> = permutation.inverse_rows().iter().copied().collect();
                assert_eq!(forward.len(), permutation.len());
                assert_eq!(inverse.len(), permutation.len());
            }
        }
    }

    #[test]
    fn legal_row_permutations_compose_for_every_ordered_transform_pair() {
        let state = game(90);
        let prelude = MarketPrelude::default();
        for first in D6Transform::ALL {
            let first_permutation = LegalActionPermutation::new(&state, &prelude, first).unwrap();
            let first_state = state.transformed(first).unwrap();
            for second in D6Transform::ALL {
                let second_permutation =
                    LegalActionPermutation::new(&first_state, &prelude, second).unwrap();
                let composed_permutation =
                    LegalActionPermutation::new(&state, &prelude, second.compose(first)).unwrap();
                for source_row in 0..first_permutation.len() {
                    assert_eq!(
                        second_permutation.forward_rows()
                            [first_permutation.forward_rows()[source_row]],
                        composed_permutation.forward_rows()[source_row]
                    );
                }
            }
        }
    }

    #[test]
    fn transition_equivariance_holds_for_every_transform() {
        let state = generated_state(73, 4);
        let action = state
            .legal_turn_actions(&MarketPrelude::default())
            .unwrap()
            .into_iter()
            .find(|action| action.wildlife.is_some())
            .unwrap();
        let next = state.transition(&action).unwrap();

        for transform in D6Transform::ALL {
            let transformed_state = state.transformed(transform).unwrap();
            let transformed_action = state.transform_turn_action(&action, transform).unwrap();
            assert_eq!(
                transformed_state.transition(&transformed_action).unwrap(),
                next.transformed(transform).unwrap()
            );
        }
    }

    #[test]
    fn independent_draft_token_semantics_are_equivariant() {
        let state = state_with_paid_prelude();
        let action = state
            .legal_turn_actions(&MarketPrelude::default())
            .unwrap()
            .into_iter()
            .find(|action| matches!(action.draft, DraftChoice::Independent { .. }))
            .expect("active nature token must enable independent drafts");
        let next = state.transition(&action).unwrap();
        assert_eq!(
            next.boards()[state.current_player()].nature_tokens() + 1,
            state.boards()[state.current_player()].nature_tokens()
        );

        for transform in D6Transform::ALL {
            let transformed_action = action.transformed(&state, transform).unwrap();
            assert!(matches!(
                transformed_action.draft,
                DraftChoice::Independent { .. }
            ));
            assert_eq!(
                state
                    .transformed(transform)
                    .unwrap()
                    .transition(&transformed_action)
                    .unwrap(),
                next.transformed(transform).unwrap()
            );
        }
    }

    #[test]
    fn free_and_paid_market_preludes_remain_equivariant() {
        let cases = [
            (
                state_with_free_three_of_a_kind(),
                MarketPrelude {
                    replace_three_of_a_kind: true,
                    wildlife_wipes: Vec::new(),
                },
            ),
            (
                state_with_paid_prelude(),
                MarketPrelude {
                    replace_three_of_a_kind: false,
                    wildlife_wipes: vec![WildlifeWipe {
                        slots: vec![MarketSlot::ZERO],
                    }],
                },
            ),
        ];

        for (state, prelude) in cases {
            let action = state
                .legal_turn_actions(&prelude)
                .unwrap()
                .into_iter()
                .next()
                .unwrap();
            assert_eq!(action.prelude(), prelude);
            let next = state.transition(&action).unwrap();
            for transform in D6Transform::ALL {
                let permutation = LegalActionPermutation::new(&state, &prelude, transform).unwrap();
                permutation.validate().unwrap();
                let transformed_action = state.transform_turn_action(&action, transform).unwrap();
                assert_eq!(transformed_action.prelude(), prelude);
                assert_eq!(
                    state
                        .transformed(transform)
                        .unwrap()
                        .transition(&transformed_action)
                        .unwrap(),
                    next.transformed(transform).unwrap()
                );
            }
        }
    }

    #[test]
    fn keystone_actions_canonicalize_single_terrain_rotation() {
        for seed in 0..2_048 {
            let state = game(50_000 + seed);
            let actions = state.legal_turn_actions(&MarketPrelude::default()).unwrap();
            let Some(action) = actions.into_iter().find(|action| {
                let tile = match action.draft {
                    DraftChoice::Paired { slot } => state.market().tiles[slot.index()],
                    DraftChoice::Independent { tile_slot, .. } => {
                        state.market().tiles[tile_slot.index()]
                    }
                };
                tile.is_some_and(|tile| tile.terrain_b.is_none())
            }) else {
                continue;
            };
            for transform in D6Transform::ALL {
                assert_eq!(
                    state
                        .transform_turn_action(&action, transform)
                        .unwrap()
                        .tile
                        .rotation,
                    Rotation::ZERO
                );
            }
            return;
        }
        panic!("expected a deterministic market containing a keystone");
    }

    #[test]
    fn policy_permutations_round_trip_compose_and_preserve_argmax_identity() {
        let state = generated_state(91, 3);
        let first = D6Transform::new(2, true).unwrap();
        let second = D6Transform::new(4, false).unwrap();
        let first_permutation =
            LegalActionPermutation::new(&state, &MarketPrelude::default(), first).unwrap();
        let first_state = state.transformed(first).unwrap();
        let second_permutation =
            LegalActionPermutation::new(&first_state, &MarketPrelude::default(), second).unwrap();
        let composed = second.compose(first);
        let composed_permutation =
            LegalActionPermutation::new(&state, &MarketPrelude::default(), composed).unwrap();

        for source_row in 0..first_permutation.len() {
            assert_eq!(
                second_permutation.forward_rows()[first_permutation.forward_rows()[source_row]],
                composed_permutation.forward_rows()[source_row]
            );
        }

        let policy: Vec<_> = (0..first_permutation.len())
            .map(|row| row as f64 / first_permutation.len() as f64)
            .collect();
        let transformed = first_permutation.permute_forward(&policy).unwrap();
        assert_eq!(
            first_permutation.permute_inverse(&transformed).unwrap(),
            policy
        );
        let twice = second_permutation.permute_forward(&transformed).unwrap();
        assert_eq!(
            twice,
            composed_permutation.permute_forward(&policy).unwrap()
        );

        let source_argmax = policy
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1))
            .unwrap()
            .0;
        let transformed_argmax = transformed
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1))
            .unwrap()
            .0;
        assert_eq!(
            transformed_argmax,
            first_permutation.forward_rows()[source_argmax]
        );
        assert_eq!(
            first_permutation.permute_forward(&policy[..policy.len() - 1]),
            Err(D6Error::PolicyLength {
                expected: policy.len(),
                actual: policy.len() - 1,
            })
        );
    }

    #[test]
    fn transformed_actions_are_recomputed_by_value() {
        let state = game(123);
        let actions = state.legal_turn_actions(&MarketPrelude::default()).unwrap();
        let transform = D6Transform::new(5, true).unwrap();
        let transformed_state = state.transformed(transform).unwrap();
        let transformed_actions = transformed_state
            .legal_turn_actions(&MarketPrelude::default())
            .unwrap();
        for action in actions.iter().step_by((actions.len() / 32).max(1)) {
            let transformed = state.transform_turn_action(action, transform).unwrap();
            assert!(transformed_actions.contains(&transformed));
        }
    }

    #[test]
    fn action_constructor_remains_compatible_with_exact_transform() {
        let state = game(777);
        let action = TurnAction::paired(
            MarketSlot::ZERO,
            state.boards()[0].frontier()[0],
            Rotation::ZERO,
        );
        let transform = D6Transform::new(3, false).unwrap();
        let transformed = action.transformed(&state, transform).unwrap();
        assert_eq!(transformed.draft, action.draft);
        assert_eq!(
            transformed.tile.coord,
            transform.transform_coord(action.tile.coord).unwrap()
        );
    }
}
