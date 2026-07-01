//! Lossless spatial encodings for the R0 representation tournament.
//!
//! Every arm carries the same six raw tile-semantic channels. Bounded arms
//! replace absolute coordinates with a local index and retain every entity
//! outside that support in an exact-coordinate overflow stream.

use std::collections::BTreeSet;

use cascadia_game::{
    D6Error, D6Transform, HexCoord, Rotation, Terrain, Tile, TileId, Wildlife, WildlifeMask,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    BOARD_ENTITY_SIZE, BOARD_SLOTS, MARKET_ENTITY_SIZE, MAX_BOARD_TILES, PositionRecord,
    RECORD_SIZE, TARGET_DIM,
};

pub const SPATIAL_REPRESENTATION_SCHEMA_VERSION: u16 = 1;
pub const SPATIAL_SEMANTIC_CHANNELS: usize = BOARD_ENTITY_SIZE - 2;
pub const SPATIAL_PACKED_MAGIC: &[u8; 8] = b"CSR0SP1\0";
pub const SPATIAL_PACKED_HEADER_SIZE: usize = 12;
pub const POSITION_NON_SPATIAL_BYTES: usize =
    RECORD_SIZE - BOARD_SLOTS * MAX_BOARD_TILES * BOARD_ENTITY_SIZE;

pub const TERRAIN_A_CHANNEL: usize = 0;
pub const TERRAIN_B_CHANNEL: usize = 1;
pub const ROTATION_CHANNEL: usize = 2;
pub const ALLOWED_WILDLIFE_CHANNEL: usize = 3;
pub const PLACED_WILDLIFE_CHANNEL: usize = 4;
pub const KEYSTONE_CHANNEL: usize = 5;

const NONE: u8 = u8::MAX;
const HISTORICAL_SQUARE_RADIUS: i8 = 10;
const HISTORICAL_SQUARE_DIM: usize = 21;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
#[repr(u8)]
pub enum SpatialArm {
    ExactEntityControl = 0,
    HexRadius6 = 1,
    HexRadius5 = 2,
    HexRadius4 = 3,
    HistoricalSquare21 = 4,
}

impl SpatialArm {
    pub const ALL: [Self; 5] = [
        Self::ExactEntityControl,
        Self::HexRadius6,
        Self::HexRadius5,
        Self::HexRadius4,
        Self::HistoricalSquare21,
    ];

    pub const fn id(self) -> &'static str {
        match self {
            Self::ExactEntityControl => "exact-entity-control",
            Self::HexRadius6 => "hex-radius-6-127",
            Self::HexRadius5 => "hex-radius-5-91",
            Self::HexRadius4 => "hex-radius-4-61",
            Self::HistoricalSquare21 => "historical-square-21x21-441",
        }
    }

    pub const fn code(self) -> u8 {
        self as u8
    }

    pub fn from_id(id: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|arm| arm.id() == id)
    }

    pub const fn from_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(Self::ExactEntityControl),
            1 => Some(Self::HexRadius6),
            2 => Some(Self::HexRadius5),
            3 => Some(Self::HexRadius4),
            4 => Some(Self::HistoricalSquare21),
            _ => None,
        }
    }

    pub const fn local_capacity(self) -> usize {
        match self {
            Self::ExactEntityControl => 0,
            Self::HexRadius6 => centered_hex_capacity(6),
            Self::HexRadius5 => centered_hex_capacity(5),
            Self::HexRadius4 => centered_hex_capacity(4),
            Self::HistoricalSquare21 => HISTORICAL_SQUARE_DIM * HISTORICAL_SQUARE_DIM,
        }
    }

    pub const fn hex_radius(self) -> Option<u8> {
        match self {
            Self::HexRadius6 => Some(6),
            Self::HexRadius5 => Some(5),
            Self::HexRadius4 => Some(4),
            Self::ExactEntityControl | Self::HistoricalSquare21 => None,
        }
    }

    pub const fn uses_recentered_frame(self) -> bool {
        matches!(self, Self::HexRadius6 | Self::HexRadius5 | Self::HexRadius4)
    }

    pub const fn local_index_bytes(self) -> usize {
        match self {
            Self::HexRadius6 | Self::HexRadius5 | Self::HexRadius4 => 1,
            Self::HistoricalSquare21 => 2,
            Self::ExactEntityControl => 0,
        }
    }

    pub fn local_index(self, relative: HexCoord) -> Option<u16> {
        match self {
            Self::ExactEntityControl => None,
            Self::HexRadius6 => hex_disk_index(6, relative),
            Self::HexRadius5 => hex_disk_index(5, relative),
            Self::HexRadius4 => hex_disk_index(4, relative),
            Self::HistoricalSquare21 => historical_square_index(relative),
        }
    }

    pub fn local_coord(self, index: u16) -> Option<HexCoord> {
        match self {
            Self::ExactEntityControl => None,
            Self::HexRadius6 => hex_disk_coord(6, index),
            Self::HexRadius5 => hex_disk_coord(5, index),
            Self::HexRadius4 => hex_disk_coord(4, index),
            Self::HistoricalSquare21 => historical_square_coord(index),
        }
    }

    /// Returns the transformed local index when the transformed coordinate
    /// remains in this arm's local support.
    ///
    /// Complete hex disks are D6-closed and therefore always return `Some`.
    /// The historical axial square is not D6-closed and can return `None`;
    /// callers must route that entity through the exact overflow stream.
    pub fn transform_local_index(
        self,
        index: u16,
        transform: D6Transform,
    ) -> Result<Option<u16>, SpatialRepresentationError> {
        let relative = self
            .local_coord(index)
            .ok_or(SpatialRepresentationError::InvalidLocalIndex { arm: self, index })?;
        let transformed = transform.transform_coord(relative)?;
        Ok(self.local_index(transformed))
    }
}

pub const fn centered_hex_capacity(radius: u8) -> usize {
    let radius = radius as usize;
    1 + 3 * radius * (radius + 1)
}

/// Exact F2 minimax integer center with a stable `(q, r)` tie-break.
///
/// D6 augmentation must transform the selected center together with the
/// entities. It must not independently rerun this tie-break after a transform.
pub fn deterministic_integer_center(coordinates: &[HexCoord]) -> HexCoord {
    if coordinates.is_empty() {
        return HexCoord::ORIGIN;
    }
    let mut min_q = i16::MAX;
    let mut max_q = i16::MIN;
    let mut min_r = i16::MAX;
    let mut max_r = i16::MIN;
    let mut min_s = i16::MAX;
    let mut max_s = i16::MIN;
    for coord in coordinates {
        let q = i16::from(coord.q);
        let r = i16::from(coord.r);
        let s = -q - r;
        min_q = min_q.min(q);
        max_q = max_q.max(q);
        min_r = min_r.min(r);
        max_r = max_r.max(r);
        min_s = min_s.min(s);
        max_s = max_s.max(s);
    }
    let lower_bound = [max_q - min_q, max_r - min_r, max_s - min_s]
        .into_iter()
        .max()
        .map_or(0, |span| (span + 1) / 2);
    for radius in lower_bound..=48 {
        let q_low = max_q - radius;
        let q_high = min_q + radius;
        let r_low = max_r - radius;
        let r_high = min_r + radius;
        let s_low = max_s - radius;
        let s_high = min_s + radius;
        if q_low > q_high
            || r_low > r_high
            || s_low > s_high
            || q_low + r_low + s_low > 0
            || q_high + r_high + s_high < 0
        {
            continue;
        }
        for q in q_low..=q_high {
            let minimum_r = r_low.max(-s_high - q);
            let maximum_r = r_high.min(-s_low - q);
            if minimum_r <= maximum_r {
                return HexCoord::new(q as i8, minimum_r as i8);
            }
        }
    }
    unreachable!("supported V2 board coordinates always have an integer minimax center")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SpatialEntity {
    pub coord: HexCoord,
    pub channels: [u8; SPATIAL_SEMANTIC_CHANNELS],
}

impl SpatialEntity {
    pub fn from_board_entity(
        board: usize,
        row: usize,
        entity: [u8; BOARD_ENTITY_SIZE],
    ) -> Result<Self, SpatialRepresentationError> {
        if entity.iter().all(|value| *value == NONE) {
            return Err(SpatialRepresentationError::InvalidEntity {
                board,
                row,
                reason: "active board row is the all-NONE padding sentinel".to_owned(),
            });
        }
        let value = Self {
            coord: HexCoord::new(entity[0] as i8, entity[1] as i8),
            channels: entity[2..]
                .try_into()
                .expect("board semantic channel width is fixed"),
        };
        value.validate(board, row)?;
        Ok(value)
    }

    pub const fn to_board_entity(self) -> [u8; BOARD_ENTITY_SIZE] {
        [
            self.coord.q as u8,
            self.coord.r as u8,
            self.channels[TERRAIN_A_CHANNEL],
            self.channels[TERRAIN_B_CHANNEL],
            self.channels[ROTATION_CHANNEL],
            self.channels[ALLOWED_WILDLIFE_CHANNEL],
            self.channels[PLACED_WILDLIFE_CHANNEL],
            self.channels[KEYSTONE_CHANNEL],
        ]
    }

    pub fn transformed(self, transform: D6Transform) -> Result<Self, SpatialRepresentationError> {
        validate_channels(self.channels).map_err(|reason| {
            SpatialRepresentationError::InvalidEntity {
                board: 0,
                row: 0,
                reason,
            }
        })?;
        let transformed_coord = transform.transform_coord(self.coord)?;
        if transformed_coord.to_index().is_none() {
            return Err(
                SpatialRepresentationError::TransformedCoordinateOutOfBounds {
                    transform,
                    source_coord: self.coord,
                    transformed: transformed_coord,
                },
            );
        }
        let terrain_a =
            terrain_from_code(self.channels[TERRAIN_A_CHANNEL]).expect("validated primary terrain");
        let terrain_b = optional_terrain_from_code(self.channels[TERRAIN_B_CHANNEL])
            .expect("validated secondary terrain");
        let tile = Tile {
            id: TileId(0),
            terrain_a,
            terrain_b,
            wildlife: WildlifeMask::from_bits(self.channels[ALLOWED_WILDLIFE_CHANNEL]),
            keystone: self.channels[KEYSTONE_CHANNEL] != 0,
        };
        let rotation =
            Rotation::new(self.channels[ROTATION_CHANNEL]).expect("validated tile rotation");
        let mut channels = self.channels;
        channels[ROTATION_CHANNEL] = transform.transform_tile_rotation(tile, rotation).get();
        Ok(Self {
            coord: transformed_coord,
            channels,
        })
    }

    fn validate(self, board: usize, row: usize) -> Result<(), SpatialRepresentationError> {
        if self.coord.to_index().is_none() {
            return Err(SpatialRepresentationError::CoordinateOutOfBounds {
                board,
                row,
                coord: self.coord,
            });
        }
        validate_channels(self.channels)
            .map_err(|reason| SpatialRepresentationError::InvalidEntity { board, row, reason })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct IndexedSpatialEntity {
    pub index: u16,
    pub channels: [u8; SPATIAL_SEMANTIC_CHANNELS],
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpatialBoardRepresentation {
    arm: SpatialArm,
    center: HexCoord,
    exact_entities: Vec<SpatialEntity>,
    local_entities: Vec<IndexedSpatialEntity>,
    overflow_entities: Vec<SpatialEntity>,
}

impl SpatialBoardRepresentation {
    pub fn from_entities(
        arm: SpatialArm,
        entities: Vec<SpatialEntity>,
    ) -> Result<Self, SpatialRepresentationError> {
        let center = if arm.uses_recentered_frame() {
            let coordinates = entities
                .iter()
                .map(|entity| entity.coord)
                .collect::<Vec<_>>();
            deterministic_integer_center(&coordinates)
        } else {
            HexCoord::ORIGIN
        };
        Self::from_entities_at_center(arm, center, entities)
    }

    pub const fn arm(&self) -> SpatialArm {
        self.arm
    }

    pub const fn center(&self) -> HexCoord {
        self.center
    }

    pub fn exact_entities(&self) -> &[SpatialEntity] {
        &self.exact_entities
    }

    pub fn local_entities(&self) -> &[IndexedSpatialEntity] {
        &self.local_entities
    }

    pub fn overflow_entities(&self) -> &[SpatialEntity] {
        &self.overflow_entities
    }

    pub fn entity_count(&self) -> usize {
        self.exact_entities.len() + self.local_entities.len() + self.overflow_entities.len()
    }

    pub fn reconstruct_entities(&self) -> Result<Vec<SpatialEntity>, SpatialRepresentationError> {
        let mut entities = Vec::with_capacity(self.entity_count());
        entities.extend(self.exact_entities.iter().copied());
        for local in &self.local_entities {
            let relative = self.arm.local_coord(local.index).ok_or(
                SpatialRepresentationError::InvalidLocalIndex {
                    arm: self.arm,
                    index: local.index,
                },
            )?;
            entities.push(SpatialEntity {
                coord: add_coords(relative, self.center)?,
                channels: local.channels,
            });
        }
        entities.extend(self.overflow_entities.iter().copied());
        entities.sort_unstable();
        ensure_unique_coordinates(&entities)?;
        Ok(entities)
    }

    pub fn transformed(&self, transform: D6Transform) -> Result<Self, SpatialRepresentationError> {
        let transformed_center = if self.arm.uses_recentered_frame() {
            let center = transform.transform_coord(self.center)?;
            if center.to_index().is_none() {
                return Err(
                    SpatialRepresentationError::TransformedCoordinateOutOfBounds {
                        transform,
                        source_coord: self.center,
                        transformed: center,
                    },
                );
            }
            center
        } else {
            HexCoord::ORIGIN
        };
        let entities = self
            .reconstruct_entities()?
            .into_iter()
            .map(|entity| entity.transformed(transform))
            .collect::<Result<Vec<_>, _>>()?;
        Self::from_entities_at_center(self.arm, transformed_center, entities)
    }

    fn from_entities_at_center(
        arm: SpatialArm,
        center: HexCoord,
        mut entities: Vec<SpatialEntity>,
    ) -> Result<Self, SpatialRepresentationError> {
        entities.sort_unstable();
        ensure_unique_coordinates(&entities)?;
        for (row, entity) in entities.iter().copied().enumerate() {
            entity.validate(0, row)?;
        }

        let mut representation = Self {
            arm,
            center,
            exact_entities: Vec::new(),
            local_entities: Vec::new(),
            overflow_entities: Vec::new(),
        };
        if arm == SpatialArm::ExactEntityControl {
            representation.exact_entities = entities;
        } else {
            for entity in entities {
                let relative = subtract_coords(entity.coord, center)?;
                if let Some(index) = arm.local_index(relative) {
                    representation.local_entities.push(IndexedSpatialEntity {
                        index,
                        channels: entity.channels,
                    });
                } else {
                    representation.overflow_entities.push(entity);
                }
            }
            representation
                .local_entities
                .sort_unstable_by_key(|entity| entity.index);
            representation.overflow_entities.sort_unstable();
        }
        representation.validate()?;
        Ok(representation)
    }

    fn validate(&self) -> Result<(), SpatialRepresentationError> {
        if self.entity_count() > MAX_BOARD_TILES {
            return Err(SpatialRepresentationError::InvalidRepresentation(
                "a board representation exceeds the 23-tile rules limit".to_owned(),
            ));
        }
        match self.arm {
            SpatialArm::ExactEntityControl => {
                if self.center != HexCoord::ORIGIN
                    || !self.local_entities.is_empty()
                    || !self.overflow_entities.is_empty()
                {
                    return Err(SpatialRepresentationError::InvalidRepresentation(
                        "exact control must use only exact-coordinate entities".to_owned(),
                    ));
                }
            }
            SpatialArm::HistoricalSquare21 => {
                if self.center != HexCoord::ORIGIN || !self.exact_entities.is_empty() {
                    return Err(SpatialRepresentationError::InvalidRepresentation(
                        "historical 21x21 support is fixed at the rules origin".to_owned(),
                    ));
                }
            }
            SpatialArm::HexRadius6 | SpatialArm::HexRadius5 | SpatialArm::HexRadius4 => {
                if !self.exact_entities.is_empty() {
                    return Err(SpatialRepresentationError::InvalidRepresentation(
                        "bounded hex arms must use local and overflow entities".to_owned(),
                    ));
                }
            }
        }

        if !strictly_increasing_by(&self.local_entities, |entity| entity.index) {
            return Err(SpatialRepresentationError::InvalidRepresentation(
                "local entity indices must be unique and sorted".to_owned(),
            ));
        }
        if !strictly_increasing_by(&self.exact_entities, |entity| entity.coord)
            || !strictly_increasing_by(&self.overflow_entities, |entity| entity.coord)
        {
            return Err(SpatialRepresentationError::InvalidRepresentation(
                "exact-coordinate entities must be unique and sorted".to_owned(),
            ));
        }
        for local in &self.local_entities {
            if self.arm.local_coord(local.index).is_none() {
                return Err(SpatialRepresentationError::InvalidLocalIndex {
                    arm: self.arm,
                    index: local.index,
                });
            }
            validate_channels(local.channels)
                .map_err(SpatialRepresentationError::InvalidRepresentation)?;
        }
        for (row, entity) in self
            .exact_entities
            .iter()
            .chain(&self.overflow_entities)
            .copied()
            .enumerate()
        {
            entity.validate(0, row)?;
        }
        for overflow in &self.overflow_entities {
            let relative = subtract_coords(overflow.coord, self.center)?;
            if self.arm.local_index(relative).is_some() {
                return Err(SpatialRepresentationError::InvalidRepresentation(
                    "overflow entity is representable by the local support".to_owned(),
                ));
            }
        }
        let reconstructed = self.reconstruct_entities()?;
        if reconstructed.len() != self.entity_count() {
            return Err(SpatialRepresentationError::InvalidRepresentation(
                "reconstruction changed the entity count".to_owned(),
            ));
        }
        Ok(())
    }

    fn packed_len(&self) -> usize {
        match self.arm {
            SpatialArm::ExactEntityControl => self.exact_entities.len() * BOARD_ENTITY_SIZE,
            SpatialArm::HexRadius6 | SpatialArm::HexRadius5 | SpatialArm::HexRadius4 => {
                3 + self.local_entities.len() * (1 + SPATIAL_SEMANTIC_CHANNELS)
                    + self.overflow_entities.len() * BOARD_ENTITY_SIZE
            }
            SpatialArm::HistoricalSquare21 => {
                1 + self.local_entities.len() * (2 + SPATIAL_SEMANTIC_CHANNELS)
                    + self.overflow_entities.len() * BOARD_ENTITY_SIZE
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpatialPositionMetadata {
    pub game_index: u64,
    pub turn: u8,
    pub active_seat: u8,
    pub player_count: u8,
    pub total_turns: u8,
    pub board_counts: [u8; BOARD_SLOTS],
    pub nature_tokens: [u8; BOARD_SLOTS],
    pub scoring_cards: [u8; 5],
    pub habitat_bonuses: bool,
    pub wildlife_counts: [[u8; 5]; BOARD_SLOTS],
    pub habitat_sizes: [[u8; 5]; BOARD_SLOTS],
    pub market_entities: [[u8; MARKET_ENTITY_SIZE]; 4],
    pub targets: [u16; TARGET_DIM],
}

impl SpatialPositionMetadata {
    fn from_record(record: &PositionRecord) -> Self {
        Self {
            game_index: record.game_index,
            turn: record.turn,
            active_seat: record.active_seat,
            player_count: record.player_count,
            total_turns: record.total_turns,
            board_counts: record.board_counts,
            nature_tokens: record.nature_tokens,
            scoring_cards: record.scoring_cards,
            habitat_bonuses: record.habitat_bonuses,
            wildlife_counts: record.wildlife_counts,
            habitat_sizes: record.habitat_sizes,
            market_entities: record.market_entities,
            targets: record.targets,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpatialRepresentationAccounting {
    pub packed_bytes: usize,
    pub packed_spatial_bytes: usize,
    pub local_capacity_rows: usize,
    pub active_local_rows: usize,
    pub exact_entity_rows: usize,
    pub overflow_entity_rows: usize,
    pub semantic_entity_rows: usize,
    pub dense_raw_scalar_slots: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpatialPositionRepresentation {
    pub arm: SpatialArm,
    pub metadata: SpatialPositionMetadata,
    pub boards: [SpatialBoardRepresentation; BOARD_SLOTS],
}

impl SpatialPositionRepresentation {
    pub fn from_record(
        record: &PositionRecord,
        arm: SpatialArm,
    ) -> Result<Self, SpatialRepresentationError> {
        validate_record_shape(record)?;
        let mut boards = Vec::with_capacity(BOARD_SLOTS);
        for board in 0..BOARD_SLOTS {
            let count = usize::from(record.board_counts[board]);
            let mut entities = Vec::with_capacity(count);
            for row in 0..count {
                entities.push(SpatialEntity::from_board_entity(
                    board,
                    row,
                    record.board_entities[board][row],
                )?);
            }
            boards.push(SpatialBoardRepresentation::from_entities(arm, entities)?);
        }
        let boards: [SpatialBoardRepresentation; BOARD_SLOTS] =
            boards.try_into().map_err(|_| {
                SpatialRepresentationError::InvalidRepresentation(
                    "board representation count changed during extraction".to_owned(),
                )
            })?;
        let representation = Self {
            arm,
            metadata: SpatialPositionMetadata::from_record(record),
            boards,
        };
        representation.validate()?;
        Ok(representation)
    }

    pub fn to_position_record(&self) -> Result<PositionRecord, SpatialRepresentationError> {
        self.validate()?;
        let mut board_entities = [[[NONE; BOARD_ENTITY_SIZE]; MAX_BOARD_TILES]; BOARD_SLOTS];
        for (board, representation) in self.boards.iter().enumerate() {
            let entities = representation.reconstruct_entities()?;
            if entities.len() != usize::from(self.metadata.board_counts[board]) {
                return Err(SpatialRepresentationError::BoardCountMismatch {
                    board,
                    expected: usize::from(self.metadata.board_counts[board]),
                    actual: entities.len(),
                });
            }
            for (row, entity) in entities.into_iter().enumerate() {
                board_entities[board][row] = entity.to_board_entity();
            }
        }
        Ok(PositionRecord {
            game_index: self.metadata.game_index,
            turn: self.metadata.turn,
            active_seat: self.metadata.active_seat,
            player_count: self.metadata.player_count,
            total_turns: self.metadata.total_turns,
            board_counts: self.metadata.board_counts,
            nature_tokens: self.metadata.nature_tokens,
            scoring_cards: self.metadata.scoring_cards,
            habitat_bonuses: self.metadata.habitat_bonuses,
            wildlife_counts: self.metadata.wildlife_counts,
            habitat_sizes: self.metadata.habitat_sizes,
            board_entities,
            market_entities: self.metadata.market_entities,
            targets: self.metadata.targets,
        })
    }

    pub fn transformed(&self, transform: D6Transform) -> Result<Self, SpatialRepresentationError> {
        let boards = self
            .boards
            .iter()
            .map(|board| board.transformed(transform))
            .collect::<Result<Vec<_>, _>>()?
            .try_into()
            .map_err(|_| {
                SpatialRepresentationError::InvalidRepresentation(
                    "board representation count changed during D6 transform".to_owned(),
                )
            })?;
        let transformed = Self {
            arm: self.arm,
            metadata: self.metadata.clone(),
            boards,
        };
        transformed.validate()?;
        Ok(transformed)
    }

    pub fn accounting(&self) -> SpatialRepresentationAccounting {
        let local_capacity_rows = self.arm.local_capacity() * BOARD_SLOTS;
        let active_local_rows = self
            .boards
            .iter()
            .map(|board| board.local_entities.len())
            .sum();
        let exact_entity_rows = self
            .boards
            .iter()
            .map(|board| board.exact_entities.len())
            .sum();
        let overflow_entity_rows = self
            .boards
            .iter()
            .map(|board| board.overflow_entities.len())
            .sum();
        let semantic_entity_rows = active_local_rows + exact_entity_rows + overflow_entity_rows;
        let packed_spatial_bytes = self
            .boards
            .iter()
            .map(SpatialBoardRepresentation::packed_len)
            .sum();
        let packed_bytes =
            SPATIAL_PACKED_HEADER_SIZE + POSITION_NON_SPATIAL_BYTES + packed_spatial_bytes;
        let dense_raw_scalar_slots = local_capacity_rows * (SPATIAL_SEMANTIC_CHANNELS + 1)
            + (exact_entity_rows + overflow_entity_rows) * BOARD_ENTITY_SIZE;
        SpatialRepresentationAccounting {
            packed_bytes,
            packed_spatial_bytes,
            local_capacity_rows,
            active_local_rows,
            exact_entity_rows,
            overflow_entity_rows,
            semantic_entity_rows,
            dense_raw_scalar_slots,
        }
    }

    pub fn to_packed_bytes(&self) -> Result<Vec<u8>, SpatialRepresentationError> {
        self.validate()?;
        let accounting = self.accounting();
        let mut output = Vec::with_capacity(accounting.packed_bytes);
        output.extend_from_slice(SPATIAL_PACKED_MAGIC);
        output.extend_from_slice(&SPATIAL_REPRESENTATION_SCHEMA_VERSION.to_le_bytes());
        output.push(self.arm.code());
        output.push(0);
        write_metadata_prefix(&mut output, &self.metadata);
        for (board_index, board) in self.boards.iter().enumerate() {
            let count = usize::from(self.metadata.board_counts[board_index]);
            match self.arm {
                SpatialArm::ExactEntityControl => {
                    debug_assert_eq!(board.exact_entities.len(), count);
                    for entity in &board.exact_entities {
                        write_exact_entity(&mut output, *entity);
                    }
                }
                SpatialArm::HexRadius6 | SpatialArm::HexRadius5 | SpatialArm::HexRadius4 => {
                    output.push(board.center.q as u8);
                    output.push(board.center.r as u8);
                    output.push(board.local_entities.len() as u8);
                    for entity in &board.local_entities {
                        output.push(entity.index as u8);
                        output.extend_from_slice(&entity.channels);
                    }
                    for entity in &board.overflow_entities {
                        write_exact_entity(&mut output, *entity);
                    }
                    debug_assert_eq!(
                        board.local_entities.len() + board.overflow_entities.len(),
                        count
                    );
                }
                SpatialArm::HistoricalSquare21 => {
                    output.push(board.local_entities.len() as u8);
                    for entity in &board.local_entities {
                        output.extend_from_slice(&entity.index.to_le_bytes());
                        output.extend_from_slice(&entity.channels);
                    }
                    for entity in &board.overflow_entities {
                        write_exact_entity(&mut output, *entity);
                    }
                    debug_assert_eq!(
                        board.local_entities.len() + board.overflow_entities.len(),
                        count
                    );
                }
            }
        }
        write_metadata_suffix(&mut output, &self.metadata);
        debug_assert_eq!(output.len(), accounting.packed_bytes);
        Ok(output)
    }

    pub fn from_packed_bytes(bytes: &[u8]) -> Result<Self, SpatialRepresentationError> {
        let mut reader = PackedReader::new(bytes);
        if reader.read_array::<8>()? != *SPATIAL_PACKED_MAGIC {
            return Err(SpatialRepresentationError::InvalidPackedMagic);
        }
        let schema_version = reader.read_u16()?;
        if schema_version != SPATIAL_REPRESENTATION_SCHEMA_VERSION {
            return Err(SpatialRepresentationError::UnsupportedPackedSchema(
                schema_version,
            ));
        }
        let arm_code = reader.read_u8()?;
        let arm = SpatialArm::from_code(arm_code)
            .ok_or(SpatialRepresentationError::InvalidArmCode(arm_code))?;
        if reader.read_u8()? != 0 {
            return Err(SpatialRepresentationError::InvalidRepresentation(
                "packed header reserved byte must be zero".to_owned(),
            ));
        }
        let mut metadata = read_metadata_prefix(&mut reader)?;
        let mut boards = Vec::with_capacity(BOARD_SLOTS);
        for board_index in 0..BOARD_SLOTS {
            let count = usize::from(metadata.board_counts[board_index]);
            if count > MAX_BOARD_TILES {
                return Err(SpatialRepresentationError::InvalidBoardCount {
                    board: board_index,
                    count,
                });
            }
            let board = match arm {
                SpatialArm::ExactEntityControl => {
                    let mut exact_entities = Vec::with_capacity(count);
                    for row in 0..count {
                        exact_entities.push(reader.read_exact_entity(board_index, row)?);
                    }
                    SpatialBoardRepresentation {
                        arm,
                        center: HexCoord::ORIGIN,
                        exact_entities,
                        local_entities: Vec::new(),
                        overflow_entities: Vec::new(),
                    }
                }
                SpatialArm::HexRadius6 | SpatialArm::HexRadius5 | SpatialArm::HexRadius4 => {
                    let center = HexCoord::new(reader.read_u8()? as i8, reader.read_u8()? as i8);
                    let local_count = usize::from(reader.read_u8()?);
                    if local_count > count {
                        return Err(SpatialRepresentationError::InvalidRepresentation(
                            "packed local row count exceeds board count".to_owned(),
                        ));
                    }
                    let mut local_entities = Vec::with_capacity(local_count);
                    for _ in 0..local_count {
                        let index = u16::from(reader.read_u8()?);
                        let channels = reader.read_array()?;
                        validate_channels(channels)
                            .map_err(SpatialRepresentationError::InvalidRepresentation)?;
                        local_entities.push(IndexedSpatialEntity { index, channels });
                    }
                    let overflow_count = count - local_count;
                    let mut overflow_entities = Vec::with_capacity(overflow_count);
                    for row in local_count..count {
                        overflow_entities.push(reader.read_exact_entity(board_index, row)?);
                    }
                    SpatialBoardRepresentation {
                        arm,
                        center,
                        exact_entities: Vec::new(),
                        local_entities,
                        overflow_entities,
                    }
                }
                SpatialArm::HistoricalSquare21 => {
                    let local_count = usize::from(reader.read_u8()?);
                    if local_count > count {
                        return Err(SpatialRepresentationError::InvalidRepresentation(
                            "packed local row count exceeds board count".to_owned(),
                        ));
                    }
                    let mut local_entities = Vec::with_capacity(local_count);
                    for _ in 0..local_count {
                        let index = reader.read_u16()?;
                        let channels = reader.read_array()?;
                        validate_channels(channels)
                            .map_err(SpatialRepresentationError::InvalidRepresentation)?;
                        local_entities.push(IndexedSpatialEntity { index, channels });
                    }
                    let overflow_count = count - local_count;
                    let mut overflow_entities = Vec::with_capacity(overflow_count);
                    for row in local_count..count {
                        overflow_entities.push(reader.read_exact_entity(board_index, row)?);
                    }
                    SpatialBoardRepresentation {
                        arm,
                        center: HexCoord::ORIGIN,
                        exact_entities: Vec::new(),
                        local_entities,
                        overflow_entities,
                    }
                }
            };
            board.validate()?;
            boards.push(board);
        }
        read_metadata_suffix(&mut reader, &mut metadata)?;
        if reader.remaining() != 0 {
            return Err(SpatialRepresentationError::TrailingPackedBytes(
                reader.remaining(),
            ));
        }
        let boards = boards.try_into().map_err(|_| {
            SpatialRepresentationError::InvalidRepresentation(
                "packed board count does not match the schema".to_owned(),
            )
        })?;
        let representation = Self {
            arm,
            metadata,
            boards,
        };
        representation.validate()?;
        Ok(representation)
    }

    fn validate(&self) -> Result<(), SpatialRepresentationError> {
        if self.metadata.player_count == 0 || usize::from(self.metadata.player_count) > BOARD_SLOTS
        {
            return Err(SpatialRepresentationError::InvalidRepresentation(
                "player count must be in the fixed board-slot range".to_owned(),
            ));
        }
        if usize::from(self.metadata.active_seat) >= usize::from(self.metadata.player_count) {
            return Err(SpatialRepresentationError::InvalidRepresentation(
                "active seat must be a valid absolute seat".to_owned(),
            ));
        }
        for (board, representation) in self.boards.iter().enumerate() {
            if representation.arm != self.arm {
                return Err(SpatialRepresentationError::InvalidRepresentation(
                    "board arm differs from the position arm".to_owned(),
                ));
            }
            representation.validate()?;
            let expected = usize::from(self.metadata.board_counts[board]);
            let actual = representation.entity_count();
            if expected != actual {
                return Err(SpatialRepresentationError::BoardCountMismatch {
                    board,
                    expected,
                    actual,
                });
            }
        }
        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum SpatialRepresentationError {
    #[error("board {board} count {count} exceeds the {MAX_BOARD_TILES}-tile limit")]
    InvalidBoardCount { board: usize, count: usize },
    #[error("board {board} row {row} is invalid: {reason}")]
    InvalidEntity {
        board: usize,
        row: usize,
        reason: String,
    },
    #[error("board {board} row {row} coordinate {coord:?} is outside the V2 rules grid")]
    CoordinateOutOfBounds {
        board: usize,
        row: usize,
        coord: HexCoord,
    },
    #[error("coordinate arithmetic overflows i8: ({q}, {r})")]
    CoordinateArithmeticOverflow { q: i16, r: i16 },
    #[error("multiple spatial entities occupy {0:?}")]
    DuplicateCoordinate(HexCoord),
    #[error("local index {index} is invalid for {arm:?}")]
    InvalidLocalIndex { arm: SpatialArm, index: u16 },
    #[error(
        "{transform:?} maps representation coordinate {source_coord:?} outside the V2 rules grid at {transformed:?}"
    )]
    TransformedCoordinateOutOfBounds {
        transform: D6Transform,
        source_coord: HexCoord,
        transformed: HexCoord,
    },
    #[error("board {board} entity count is {actual}; expected {expected}")]
    BoardCountMismatch {
        board: usize,
        expected: usize,
        actual: usize,
    },
    #[error("invalid spatial representation: {0}")]
    InvalidRepresentation(String),
    #[error("packed spatial representation has invalid magic")]
    InvalidPackedMagic,
    #[error("packed spatial representation schema {0} is unsupported")]
    UnsupportedPackedSchema(u16),
    #[error("packed spatial representation arm code {0} is invalid")]
    InvalidArmCode(u8),
    #[error("packed spatial representation ended unexpectedly")]
    UnexpectedPackedEnd,
    #[error("packed spatial representation has {0} trailing bytes")]
    TrailingPackedBytes(usize),
    #[error(transparent)]
    D6(#[from] D6Error),
}

fn validate_record_shape(record: &PositionRecord) -> Result<(), SpatialRepresentationError> {
    for board in 0..BOARD_SLOTS {
        let count = usize::from(record.board_counts[board]);
        if count > MAX_BOARD_TILES {
            return Err(SpatialRepresentationError::InvalidBoardCount { board, count });
        }
        for row in count..MAX_BOARD_TILES {
            if record.board_entities[board][row]
                .iter()
                .any(|value| *value != NONE)
            {
                return Err(SpatialRepresentationError::InvalidEntity {
                    board,
                    row,
                    reason: "padding row contains non-NONE data".to_owned(),
                });
            }
        }
    }
    Ok(())
}

fn validate_channels(channels: [u8; SPATIAL_SEMANTIC_CHANNELS]) -> Result<(), String> {
    let terrain_a = terrain_from_code(channels[TERRAIN_A_CHANNEL])
        .ok_or_else(|| "primary terrain code is outside [0, 4]".to_owned())?;
    let terrain_b = optional_terrain_from_code(channels[TERRAIN_B_CHANNEL])
        .ok_or_else(|| "secondary terrain code is neither NONE nor [0, 4]".to_owned())?;
    let rotation = Rotation::new(channels[ROTATION_CHANNEL])
        .ok_or_else(|| "rotation code is outside [0, 5]".to_owned())?;
    if terrain_b.is_none() && rotation != Rotation::ZERO {
        return Err("single-terrain tile rotation is not canonical zero".to_owned());
    }
    if channels[ALLOWED_WILDLIFE_CHANNEL] & !0b1_1111 != 0 {
        return Err("allowed-wildlife mask uses bits outside the five species".to_owned());
    }
    let placed = optional_wildlife_from_code(channels[PLACED_WILDLIFE_CHANNEL])
        .ok_or_else(|| "placed wildlife code is neither NONE nor [0, 4]".to_owned())?;
    if let Some(wildlife) = placed
        && !WildlifeMask::from_bits(channels[ALLOWED_WILDLIFE_CHANNEL]).contains(wildlife)
    {
        return Err("placed wildlife is absent from the tile compatibility mask".to_owned());
    }
    if channels[KEYSTONE_CHANNEL] > 1 {
        return Err("keystone channel must be zero or one".to_owned());
    }
    if channels[KEYSTONE_CHANNEL] != 0 && terrain_b.is_some() {
        return Err("keystone tile cannot have a secondary terrain".to_owned());
    }
    let _ = terrain_a;
    Ok(())
}

const fn terrain_from_code(code: u8) -> Option<Terrain> {
    match code {
        0 => Some(Terrain::Mountain),
        1 => Some(Terrain::Forest),
        2 => Some(Terrain::Prairie),
        3 => Some(Terrain::Wetland),
        4 => Some(Terrain::River),
        _ => None,
    }
}

const fn optional_terrain_from_code(code: u8) -> Option<Option<Terrain>> {
    if code == NONE {
        Some(None)
    } else {
        match terrain_from_code(code) {
            Some(terrain) => Some(Some(terrain)),
            None => None,
        }
    }
}

const fn optional_wildlife_from_code(code: u8) -> Option<Option<Wildlife>> {
    match code {
        NONE => Some(None),
        0 => Some(Some(Wildlife::Bear)),
        1 => Some(Some(Wildlife::Elk)),
        2 => Some(Some(Wildlife::Salmon)),
        3 => Some(Some(Wildlife::Hawk)),
        4 => Some(Some(Wildlife::Fox)),
        _ => None,
    }
}

fn ensure_unique_coordinates(entities: &[SpatialEntity]) -> Result<(), SpatialRepresentationError> {
    let mut coordinates = BTreeSet::new();
    for entity in entities {
        if !coordinates.insert(entity.coord) {
            return Err(SpatialRepresentationError::DuplicateCoordinate(
                entity.coord,
            ));
        }
    }
    Ok(())
}

fn strictly_increasing_by<T, K: Ord + Copy>(values: &[T], key: impl Fn(&T) -> K) -> bool {
    values.windows(2).all(|pair| key(&pair[0]) < key(&pair[1]))
}

fn subtract_coords(
    coord: HexCoord,
    center: HexCoord,
) -> Result<HexCoord, SpatialRepresentationError> {
    checked_coord(
        i16::from(coord.q) - i16::from(center.q),
        i16::from(coord.r) - i16::from(center.r),
    )
}

fn add_coords(
    relative: HexCoord,
    center: HexCoord,
) -> Result<HexCoord, SpatialRepresentationError> {
    checked_coord(
        i16::from(relative.q) + i16::from(center.q),
        i16::from(relative.r) + i16::from(center.r),
    )
}

fn checked_coord(q: i16, r: i16) -> Result<HexCoord, SpatialRepresentationError> {
    let q_i8 = i8::try_from(q)
        .map_err(|_| SpatialRepresentationError::CoordinateArithmeticOverflow { q, r })?;
    let r_i8 = i8::try_from(r)
        .map_err(|_| SpatialRepresentationError::CoordinateArithmeticOverflow { q, r })?;
    Ok(HexCoord::new(q_i8, r_i8))
}

fn hex_disk_index(radius: u8, coord: HexCoord) -> Option<u16> {
    if coord.distance(HexCoord::ORIGIN) > radius {
        return None;
    }
    let radius = i16::from(radius);
    let target_q = i16::from(coord.q);
    let target_r = i16::from(coord.r);
    let mut index = 0usize;
    for q in -radius..target_q {
        let (r_low, r_high) = axial_r_bounds(radius, q);
        index += usize::try_from(r_high - r_low + 1).expect("disk row length is positive");
    }
    let (r_low, r_high) = axial_r_bounds(radius, target_q);
    if !(r_low..=r_high).contains(&target_r) {
        return None;
    }
    index += usize::try_from(target_r - r_low).expect("disk offset is nonnegative");
    u16::try_from(index).ok()
}

fn hex_disk_coord(radius: u8, index: u16) -> Option<HexCoord> {
    if usize::from(index) >= centered_hex_capacity(radius) {
        return None;
    }
    let radius = i16::from(radius);
    let mut remaining = usize::from(index);
    for q in -radius..=radius {
        let (r_low, r_high) = axial_r_bounds(radius, q);
        let row_len = usize::try_from(r_high - r_low + 1).expect("disk row length is positive");
        if remaining < row_len {
            return Some(HexCoord::new(q as i8, (r_low + remaining as i16) as i8));
        }
        remaining -= row_len;
    }
    None
}

const fn axial_r_bounds(radius: i16, q: i16) -> (i16, i16) {
    let low_from_s = -q - radius;
    let high_from_s = -q + radius;
    let low = if -radius > low_from_s {
        -radius
    } else {
        low_from_s
    };
    let high = if radius < high_from_s {
        radius
    } else {
        high_from_s
    };
    (low, high)
}

fn historical_square_index(coord: HexCoord) -> Option<u16> {
    if !(-HISTORICAL_SQUARE_RADIUS..=HISTORICAL_SQUARE_RADIUS).contains(&coord.q)
        || !(-HISTORICAL_SQUARE_RADIUS..=HISTORICAL_SQUARE_RADIUS).contains(&coord.r)
    {
        return None;
    }
    let q = usize::try_from(i16::from(coord.q) + i16::from(HISTORICAL_SQUARE_RADIUS)).ok()?;
    let r = usize::try_from(i16::from(coord.r) + i16::from(HISTORICAL_SQUARE_RADIUS)).ok()?;
    u16::try_from(q * HISTORICAL_SQUARE_DIM + r).ok()
}

fn historical_square_coord(index: u16) -> Option<HexCoord> {
    let index = usize::from(index);
    if index >= HISTORICAL_SQUARE_DIM * HISTORICAL_SQUARE_DIM {
        return None;
    }
    Some(HexCoord::new(
        (index / HISTORICAL_SQUARE_DIM) as i8 - HISTORICAL_SQUARE_RADIUS,
        (index % HISTORICAL_SQUARE_DIM) as i8 - HISTORICAL_SQUARE_RADIUS,
    ))
}

fn write_metadata_prefix(output: &mut Vec<u8>, metadata: &SpatialPositionMetadata) {
    output.extend_from_slice(&metadata.game_index.to_le_bytes());
    output.extend_from_slice(&[
        metadata.turn,
        metadata.active_seat,
        metadata.player_count,
        metadata.total_turns,
    ]);
    output.extend_from_slice(&metadata.board_counts);
    output.extend_from_slice(&metadata.nature_tokens);
    output.extend_from_slice(&metadata.scoring_cards);
    output.push(u8::from(metadata.habitat_bonuses));
    output.extend_from_slice(&[0; 6]);
    for counts in metadata.wildlife_counts {
        output.extend_from_slice(&counts);
    }
    for sizes in metadata.habitat_sizes {
        output.extend_from_slice(&sizes);
    }
}

fn write_metadata_suffix(output: &mut Vec<u8>, metadata: &SpatialPositionMetadata) {
    for entity in metadata.market_entities {
        output.extend_from_slice(&entity);
    }
    for target in metadata.targets {
        output.extend_from_slice(&target.to_le_bytes());
    }
    output.extend_from_slice(&[0; 2]);
}

fn read_metadata_prefix(
    reader: &mut PackedReader<'_>,
) -> Result<SpatialPositionMetadata, SpatialRepresentationError> {
    let game_index = reader.read_u64()?;
    let turn = reader.read_u8()?;
    let active_seat = reader.read_u8()?;
    let player_count = reader.read_u8()?;
    let total_turns = reader.read_u8()?;
    let board_counts = reader.read_array()?;
    let nature_tokens = reader.read_array()?;
    let scoring_cards = reader.read_array()?;
    let habitat_bonuses = reader.read_u8()? != 0;
    if reader.read_array::<6>()? != [0; 6] {
        return Err(SpatialRepresentationError::InvalidRepresentation(
            "packed metadata prefix reserved bytes must be zero".to_owned(),
        ));
    }
    let mut wildlife_counts = [[0; 5]; BOARD_SLOTS];
    for counts in &mut wildlife_counts {
        *counts = reader.read_array()?;
    }
    let mut habitat_sizes = [[0; 5]; BOARD_SLOTS];
    for sizes in &mut habitat_sizes {
        *sizes = reader.read_array()?;
    }
    Ok(SpatialPositionMetadata {
        game_index,
        turn,
        active_seat,
        player_count,
        total_turns,
        board_counts,
        nature_tokens,
        scoring_cards,
        habitat_bonuses,
        wildlife_counts,
        habitat_sizes,
        market_entities: [[NONE; MARKET_ENTITY_SIZE]; 4],
        targets: [0; TARGET_DIM],
    })
}

fn read_metadata_suffix(
    reader: &mut PackedReader<'_>,
    metadata: &mut SpatialPositionMetadata,
) -> Result<(), SpatialRepresentationError> {
    for entity in &mut metadata.market_entities {
        *entity = reader.read_array()?;
    }
    for target in &mut metadata.targets {
        *target = reader.read_u16()?;
    }
    if reader.read_array::<2>()? != [0; 2] {
        return Err(SpatialRepresentationError::InvalidRepresentation(
            "packed metadata suffix reserved bytes must be zero".to_owned(),
        ));
    }
    Ok(())
}

fn write_exact_entity(output: &mut Vec<u8>, entity: SpatialEntity) {
    output.push(entity.coord.q as u8);
    output.push(entity.coord.r as u8);
    output.extend_from_slice(&entity.channels);
}

struct PackedReader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> PackedReader<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.offset)
    }

    fn read_u8(&mut self) -> Result<u8, SpatialRepresentationError> {
        Ok(self.read_array::<1>()?[0])
    }

    fn read_u16(&mut self) -> Result<u16, SpatialRepresentationError> {
        Ok(u16::from_le_bytes(self.read_array()?))
    }

    fn read_u64(&mut self) -> Result<u64, SpatialRepresentationError> {
        Ok(u64::from_le_bytes(self.read_array()?))
    }

    fn read_array<const N: usize>(&mut self) -> Result<[u8; N], SpatialRepresentationError> {
        let end = self
            .offset
            .checked_add(N)
            .ok_or(SpatialRepresentationError::UnexpectedPackedEnd)?;
        let value = self
            .bytes
            .get(self.offset..end)
            .ok_or(SpatialRepresentationError::UnexpectedPackedEnd)?
            .try_into()
            .expect("packed slice length is checked");
        self.offset = end;
        Ok(value)
    }

    fn read_exact_entity(
        &mut self,
        board: usize,
        row: usize,
    ) -> Result<SpatialEntity, SpatialRepresentationError> {
        let coord = HexCoord::new(self.read_u8()? as i8, self.read_u8()? as i8);
        let channels = self.read_array()?;
        let entity = SpatialEntity { coord, channels };
        entity.validate(board, row)?;
        Ok(entity)
    }
}

#[cfg(test)]
mod tests {
    use cascadia_game::{Board, STANDARD_TILES};
    use cascadia_game::{GameConfig, GameSeed, GameState};

    use super::*;

    fn sample_record() -> PositionRecord {
        let game = GameState::new(
            GameConfig::research_aaaaa(4).unwrap(),
            GameSeed::from_u64(9135),
        )
        .unwrap();
        PositionRecord::observe(&game, 91)
    }

    fn record_with_board_coordinates(mut coordinates: Vec<HexCoord>) -> PositionRecord {
        let mut record = sample_record();
        set_board_coordinates(&mut record, 0, &mut coordinates);
        record
    }

    fn set_board_coordinates(
        record: &mut PositionRecord,
        board: usize,
        coordinates: &mut [HexCoord],
    ) {
        coordinates.sort_unstable();
        record.board_counts[board] = coordinates.len() as u8;
        record.board_entities[board] = [[NONE; BOARD_ENTITY_SIZE]; MAX_BOARD_TILES];
        for (row, coord) in coordinates.iter().copied().enumerate() {
            record.board_entities[board][row] = [
                coord.q as u8,
                coord.r as u8,
                Terrain::Forest as u8,
                Terrain::River as u8,
                (row % 6) as u8,
                0b1_1111,
                NONE,
                0,
            ];
        }
    }

    fn assert_legal_elongated_board(coordinates: &[HexCoord]) {
        let mut board = Board::empty();
        for (index, coord) in coordinates.iter().copied().enumerate() {
            board
                .place_tile(coord, STANDARD_TILES[index], Rotation::ZERO)
                .unwrap();
        }
        board.validate().unwrap();
        assert_eq!(board.tile_count(), 23);
    }

    #[test]
    fn arm_capacities_and_indices_are_exact() {
        assert_eq!(centered_hex_capacity(4), 61);
        assert_eq!(centered_hex_capacity(5), 91);
        assert_eq!(centered_hex_capacity(6), 127);
        assert!((0..=64).all(|radius| centered_hex_capacity(radius) != 121));
        assert_eq!(SpatialArm::HistoricalSquare21.local_capacity(), 441);
        for arm in SpatialArm::ALL {
            assert_eq!(SpatialArm::from_id(arm.id()), Some(arm));
        }
        assert_eq!(SpatialArm::from_id("radius-six"), None);

        for arm in SpatialArm::ALL {
            for index in 0..arm.local_capacity() {
                let index = index as u16;
                let coord = arm.local_coord(index).unwrap();
                assert_eq!(arm.local_index(coord), Some(index));
            }
        }
    }

    #[test]
    fn complete_hex_disks_have_exact_d6_row_permutations() {
        for arm in [
            SpatialArm::HexRadius6,
            SpatialArm::HexRadius5,
            SpatialArm::HexRadius4,
        ] {
            for index in 0..arm.local_capacity() {
                let source = arm.local_coord(index as u16).unwrap();
                for transform in D6Transform::ALL {
                    let transformed_index = arm
                        .transform_local_index(index as u16, transform)
                        .unwrap()
                        .unwrap();
                    assert_eq!(
                        arm.local_coord(transformed_index).unwrap(),
                        transform.transform_coord(source).unwrap()
                    );
                }
            }
        }
    }

    #[test]
    fn deterministic_recentering_matches_f2_tie_break() {
        assert_eq!(deterministic_integer_center(&[]), HexCoord::ORIGIN);
        assert_eq!(
            deterministic_integer_center(&[HexCoord::new(10, 0), HexCoord::new(11, 0)]),
            HexCoord::new(10, 0)
        );
        let line = (10..=14).map(|q| HexCoord::new(q, -3)).collect::<Vec<_>>();
        assert_eq!(deterministic_integer_center(&line), HexCoord::new(12, -3));
    }

    #[test]
    fn every_arm_round_trips_position_and_packed_bytes() {
        let record = sample_record();
        for arm in SpatialArm::ALL {
            let representation = SpatialPositionRepresentation::from_record(&record, arm).unwrap();
            assert_eq!(representation.to_position_record().unwrap(), record);
            let packed = representation.to_packed_bytes().unwrap();
            assert_eq!(packed.len(), representation.accounting().packed_bytes);
            let decoded = SpatialPositionRepresentation::from_packed_bytes(&packed).unwrap();
            assert_eq!(decoded, representation);
            assert_eq!(decoded.to_position_record().unwrap(), record);
            assert_eq!(
                representation.accounting().semantic_entity_rows,
                record
                    .board_counts
                    .iter()
                    .map(|count| usize::from(*count))
                    .sum::<usize>()
            );
        }
    }

    #[test]
    fn legal_straight_and_bent_far_coordinate_cases_never_clip() {
        let straight = (-11..=11).map(|q| HexCoord::new(q, 0)).collect::<Vec<_>>();
        let mut bent = (0..=11).map(|q| HexCoord::new(q, 0)).collect::<Vec<_>>();
        bent.extend((1..=11).map(|r| HexCoord::new(11, r)));

        for coordinates in [straight, bent] {
            assert_legal_elongated_board(&coordinates);
            let record = record_with_board_coordinates(coordinates);
            for arm in SpatialArm::ALL {
                let representation =
                    SpatialPositionRepresentation::from_record(&record, arm).unwrap();
                assert_eq!(representation.to_position_record().unwrap(), record);
                assert_eq!(representation.boards[0].entity_count(), 23);
            }
            let radius_six =
                SpatialPositionRepresentation::from_record(&record, SpatialArm::HexRadius6)
                    .unwrap();
            assert!(!radius_six.boards[0].overflow_entities().is_empty());
        }
    }

    #[test]
    fn tied_recenter_frame_is_transformed_instead_of_reselected() {
        let record =
            record_with_board_coordinates(vec![HexCoord::new(10, 0), HexCoord::new(11, 0)]);
        let representation =
            SpatialPositionRepresentation::from_record(&record, SpatialArm::HexRadius5).unwrap();
        assert_eq!(representation.boards[0].center(), HexCoord::new(10, 0));
        let transform = D6Transform::ALL[2];
        let transformed = representation.transformed(transform).unwrap();
        assert_eq!(
            transformed.boards[0].center(),
            transform.transform_coord(HexCoord::new(10, 0)).unwrap()
        );
        assert_eq!(
            transformed
                .transformed(transform.inverse())
                .unwrap()
                .to_position_record()
                .unwrap(),
            record
        );
    }

    #[test]
    fn d6_transform_round_trip_preserves_every_arm_and_semantic_rotation() {
        let mut record = sample_record();
        for board in 0..BOARD_SLOTS {
            let mut coordinates = (-2..=2).map(|q| HexCoord::new(q, 0)).collect::<Vec<_>>();
            set_board_coordinates(&mut record, board, &mut coordinates);
        }
        let exact =
            SpatialPositionRepresentation::from_record(&record, SpatialArm::ExactEntityControl)
                .unwrap();
        for transform in D6Transform::ALL {
            let expected = exact
                .transformed(transform)
                .unwrap()
                .to_position_record()
                .unwrap();
            for arm in SpatialArm::ALL {
                let representation =
                    SpatialPositionRepresentation::from_record(&record, arm).unwrap();
                let transformed = representation.transformed(transform).unwrap();
                assert_eq!(transformed.to_position_record().unwrap(), expected);
                assert_eq!(
                    transformed
                        .transformed(transform.inverse())
                        .unwrap()
                        .to_position_record()
                        .unwrap(),
                    record
                );
                if arm.uses_recentered_frame() {
                    let freshly_extracted =
                        SpatialPositionRepresentation::from_record(&expected, arm).unwrap();
                    assert_eq!(freshly_extracted, transformed);
                }
            }
        }
    }

    #[test]
    fn historical_square_routes_nonclosed_d6_rows_through_overflow() {
        let record = record_with_board_coordinates(vec![HexCoord::new(10, 10)]);
        let representation =
            SpatialPositionRepresentation::from_record(&record, SpatialArm::HistoricalSquare21)
                .unwrap();
        assert_eq!(representation.boards[0].local_entities().len(), 1);
        let rotated = representation.transformed(D6Transform::ALL[1]).unwrap();
        assert_eq!(rotated.boards[0].local_entities().len(), 0);
        assert_eq!(rotated.boards[0].overflow_entities().len(), 1);
        assert_eq!(
            rotated
                .transformed(D6Transform::ALL[1].inverse())
                .unwrap()
                .to_position_record()
                .unwrap(),
            record
        );
    }

    #[test]
    fn packed_decoder_rejects_trailing_and_truncated_data() {
        let representation =
            SpatialPositionRepresentation::from_record(&sample_record(), SpatialArm::HexRadius5)
                .unwrap();
        let mut packed = representation.to_packed_bytes().unwrap();
        let truncated = &packed[..packed.len() - 1];
        assert!(matches!(
            SpatialPositionRepresentation::from_packed_bytes(truncated),
            Err(SpatialRepresentationError::UnexpectedPackedEnd)
        ));
        packed.push(0);
        assert!(matches!(
            SpatialPositionRepresentation::from_packed_bytes(&packed),
            Err(SpatialRepresentationError::TrailingPackedBytes(1))
        ));
    }
}
