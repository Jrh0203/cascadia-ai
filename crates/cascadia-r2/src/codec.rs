use cascadia_data::{
    BOARD_ENTITY_SIZE, BOARD_SLOTS, MARKET_ENTITY_SIZE, MAX_BOARD_TILES, PositionRecord, TARGET_DIM,
};
use cascadia_game::WildlifeMask;

use crate::{
    AxialCoord, R2Error, Result, SparsePublicState, SuppliedTile,
    model::{optional_terrain_from_code, terrain_from_code},
};

pub const PACKED_MAGIC: &[u8; 8] = b"CSR2SP1\0";
pub const PACKED_SCHEMA_VERSION: u16 = 1;

const FLAG_SUPPLIED_TILE: u16 = 1;
const NONE: u8 = u8::MAX;

impl SparsePublicState {
    pub fn to_packed_bytes(&self) -> Result<Vec<u8>> {
        let record = self.reconstruct_position_record([0; TARGET_DIM])?;
        let mut output = Vec::with_capacity(128 + self.occupied_tiles.len() * 8);
        output.extend_from_slice(PACKED_MAGIC);
        output.extend_from_slice(&PACKED_SCHEMA_VERSION.to_le_bytes());
        let flags = u16::from(self.supplied_tile.is_some()) * FLAG_SUPPLIED_TILE;
        output.extend_from_slice(&flags.to_le_bytes());

        output.extend_from_slice(&self.global.game_index.to_le_bytes());
        output.extend_from_slice(&[
            self.global.turn,
            self.global.perspective_absolute_seat,
            self.global.player_count,
            self.global.total_turns,
        ]);
        output.extend_from_slice(&self.global.scoring_cards);
        output.push(u8::from(self.global.habitat_bonuses));

        for player in &self.players {
            output.push(player.nature_tokens);
            output.push(player.occupied_count);
            output.extend_from_slice(&player.wildlife_counts);
            output.extend_from_slice(&player.largest_habitats);
            for tile in self
                .occupied_tiles
                .iter()
                .filter(|tile| tile.relative_seat == player.relative_seat)
            {
                write_signed_varint(&mut output, tile.coord.q);
                write_signed_varint(&mut output, tile.coord.r);
                output.extend_from_slice(&tile.semantic_bytes());
            }
        }

        for entity in record.market_entities {
            output.extend_from_slice(&entity);
        }
        if let Some(tile) = self.supplied_tile {
            output.extend_from_slice(&[
                tile.terrain_a as u8,
                tile.terrain_b.map_or(NONE, |terrain| terrain as u8),
                tile.wildlife_eligibility.bits(),
                u8::from(tile.keystone),
            ]);
        }
        Ok(output)
    }

    pub fn from_packed_bytes(bytes: &[u8]) -> Result<Self> {
        let mut cursor = ByteCursor::new(bytes);
        if cursor.take_array::<8>()? != *PACKED_MAGIC {
            return Err(R2Error::InvalidPackedMagic);
        }
        let schema = u16::from_le_bytes(cursor.take_array()?);
        if schema != PACKED_SCHEMA_VERSION {
            return Err(R2Error::UnsupportedPackedSchema(schema));
        }
        let flags = u16::from_le_bytes(cursor.take_array()?);
        if flags & !FLAG_SUPPLIED_TILE != 0 {
            return Err(R2Error::UnsupportedPackedFlags(flags));
        }

        let game_index = u64::from_le_bytes(cursor.take_array()?);
        let [turn, active_seat, player_count, total_turns] = cursor.take_array()?;
        if !(1..=BOARD_SLOTS as u8).contains(&player_count) {
            return Err(R2Error::NonCanonicalPacked(format!(
                "player count {player_count} is outside [1, {BOARD_SLOTS}]"
            )));
        }
        let scoring_cards = cursor.take_array()?;
        let habitat_bonuses = match cursor.take_u8()? {
            0 => false,
            1 => true,
            value => {
                return Err(R2Error::NonCanonicalPacked(format!(
                    "habitat-bonus flag {value} is not zero or one"
                )));
            }
        };

        let mut record = PositionRecord {
            game_index,
            turn,
            active_seat,
            player_count,
            total_turns,
            board_counts: [0; BOARD_SLOTS],
            nature_tokens: [0; BOARD_SLOTS],
            scoring_cards,
            habitat_bonuses,
            wildlife_counts: [[0; 5]; BOARD_SLOTS],
            habitat_sizes: [[0; 5]; BOARD_SLOTS],
            board_entities: [[[NONE; BOARD_ENTITY_SIZE]; MAX_BOARD_TILES]; BOARD_SLOTS],
            market_entities: [[NONE; MARKET_ENTITY_SIZE]; 4],
            targets: [0; TARGET_DIM],
        };

        for relative_seat in 0..player_count {
            let seat = usize::from(relative_seat);
            record.nature_tokens[seat] = cursor.take_u8()?;
            let occupied_count = cursor.take_u8()?;
            if usize::from(occupied_count) > MAX_BOARD_TILES {
                return Err(R2Error::NonCanonicalPacked(format!(
                    "relative seat {relative_seat} occupied count {occupied_count} exceeds {MAX_BOARD_TILES}"
                )));
            }
            record.board_counts[seat] = occupied_count;
            record.wildlife_counts[seat] = cursor.take_array()?;
            record.habitat_sizes[seat] = cursor.take_array()?;
            let mut previous = None;
            for row in 0..usize::from(occupied_count) {
                let coord =
                    AxialCoord::new(cursor.take_signed_varint()?, cursor.take_signed_varint()?);
                if previous.is_some_and(|previous| previous >= coord) {
                    return Err(R2Error::NonCanonicalPacked(format!(
                        "relative seat {relative_seat} occupied coordinates are not strictly ordered"
                    )));
                }
                previous = Some(coord);
                let semantic = cursor.take_array::<6>()?;
                let tile =
                    crate::OccupiedTileToken::from_semantic_bytes(relative_seat, coord, semantic)?;
                record.board_entities[seat][row] = tile.to_entity()?;
            }
        }
        for entity in &mut record.market_entities {
            *entity = cursor.take_array()?;
        }
        let supplied_tile = if flags & FLAG_SUPPLIED_TILE != 0 {
            let terrain_a_code = cursor.take_u8()?;
            let terrain_b_code = cursor.take_u8()?;
            let wildlife_mask = cursor.take_u8()?;
            let keystone = match cursor.take_u8()? {
                0 => false,
                1 => true,
                value => {
                    return Err(R2Error::NonCanonicalPacked(format!(
                        "supplied-tile keystone flag {value} is not zero or one"
                    )));
                }
            };
            let terrain_a = terrain_from_code(terrain_a_code).ok_or_else(|| {
                R2Error::NonCanonicalPacked("supplied-tile primary terrain is invalid".to_owned())
            })?;
            let terrain_b = optional_terrain_from_code(terrain_b_code).ok_or_else(|| {
                R2Error::NonCanonicalPacked("supplied-tile secondary terrain is invalid".to_owned())
            })?;
            let wildlife_eligibility = WildlifeMask::from_bits(wildlife_mask);
            if wildlife_eligibility.bits() != wildlife_mask {
                return Err(R2Error::NonCanonicalPacked(
                    "supplied-tile wildlife mask is invalid".to_owned(),
                ));
            }
            let tile = SuppliedTile {
                terrain_a,
                terrain_b,
                wildlife_eligibility,
                keystone,
            };
            tile.validate()?;
            Some(tile)
        } else {
            None
        };
        if cursor.remaining() != 0 {
            return Err(R2Error::TrailingPackedBytes(cursor.remaining()));
        }

        let state = Self::from_position_record(&record, supplied_tile)?;
        if state.to_packed_bytes()? != bytes {
            return Err(R2Error::NonCanonicalPacked(
                "decode followed by encode changed the byte stream".to_owned(),
            ));
        }
        Ok(state)
    }

    pub fn canonical_blake3(&self) -> Result<String> {
        Ok(blake3::hash(&self.to_packed_bytes()?).to_hex().to_string())
    }
}

fn write_signed_varint(output: &mut Vec<u8>, value: i16) {
    let signed = i32::from(value);
    let mut zigzag = ((signed << 1) ^ (signed >> 15)) as u16;
    loop {
        let mut byte = (zigzag & 0x7f) as u8;
        zigzag >>= 7;
        if zigzag != 0 {
            byte |= 0x80;
        }
        output.push(byte);
        if zigzag == 0 {
            break;
        }
    }
}

struct ByteCursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> ByteCursor<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn take_u8(&mut self) -> Result<u8> {
        if self.offset >= self.bytes.len() {
            return Err(R2Error::UnexpectedPackedEnd);
        }
        let value = self.bytes[self.offset];
        self.offset += 1;
        Ok(value)
    }

    fn take_array<const N: usize>(&mut self) -> Result<[u8; N]> {
        if self.offset + N > self.bytes.len() {
            return Err(R2Error::UnexpectedPackedEnd);
        }
        let value = self.bytes[self.offset..self.offset + N]
            .try_into()
            .expect("slice length was checked");
        self.offset += N;
        Ok(value)
    }

    fn take_signed_varint(&mut self) -> Result<i16> {
        let start = self.offset;
        let mut value = 0u32;
        let mut shift = 0;
        loop {
            if shift >= 21 {
                return Err(R2Error::VarintOverflow);
            }
            let byte = self.take_u8()?;
            value |= u32::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                break;
            }
            shift += 7;
        }
        if value > u32::from(u16::MAX) {
            return Err(R2Error::VarintOverflow);
        }
        let decoded = ((value >> 1) as i32) ^ -((value & 1) as i32);
        let decoded = i16::try_from(decoded).map_err(|_| R2Error::VarintOverflow)?;
        let mut canonical = Vec::new();
        write_signed_varint(&mut canonical, decoded);
        if self.bytes[start..self.offset] != canonical {
            return Err(R2Error::NonCanonicalVarint);
        }
        Ok(decoded)
    }

    fn remaining(&self) -> usize {
        self.bytes.len() - self.offset
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    proptest! {
        #[test]
        fn signed_varint_round_trips_every_i16(value in any::<i16>()) {
            let mut bytes = Vec::new();
            write_signed_varint(&mut bytes, value);
            let mut cursor = ByteCursor::new(&bytes);
            prop_assert_eq!(cursor.take_signed_varint().unwrap(), value);
            prop_assert_eq!(cursor.remaining(), 0);
        }
    }

    #[test]
    fn signed_varint_rejects_overlong_encoding() {
        let mut cursor = ByteCursor::new(&[0x80, 0x00]);
        assert!(matches!(
            cursor.take_signed_varint(),
            Err(R2Error::NonCanonicalVarint)
        ));
    }
}
