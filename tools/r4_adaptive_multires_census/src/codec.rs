use cascadia_data::{
    BOARD_ENTITY_SIZE, BOARD_SLOTS, MARKET_ENTITY_SIZE, MAX_BOARD_TILES, PositionRecord, TARGET_DIM,
};
use r2_sparse_entity_census::{AxialCoord, SparsePublicState, SuppliedTile};

use crate::{
    AdaptiveMultiResolutionState, NearFieldRadius, R4Error, Result,
    model::{hex_disk_coord, market_to_record_entity, occupied_to_record_entity},
};

pub const PACKED_MAGIC: &[u8; 8] = b"CSR4AM1\0";
pub const PACKED_SCHEMA_VERSION: u16 = 1;

const FLAG_SUPPLIED_TILE: u16 = 1;
const NONE: u8 = u8::MAX;

impl AdaptiveMultiResolutionState {
    pub fn to_packed_bytes(&self) -> Result<Vec<u8>> {
        self.validate()?;
        let mut output = Vec::with_capacity(160 + self.boards.len() * 192);
        output.extend_from_slice(PACKED_MAGIC);
        output.extend_from_slice(&PACKED_SCHEMA_VERSION.to_le_bytes());
        let flags = u16::from(self.supplied_tile.is_some()) * FLAG_SUPPLIED_TILE;
        output.extend_from_slice(&flags.to_le_bytes());
        output.push(self.radius.code());
        output.push(0);
        output.extend_from_slice(&self.global.game_index.to_le_bytes());
        output.extend_from_slice(&[
            self.global.turn,
            self.global.perspective_absolute_seat,
            self.global.player_count,
            self.global.total_turns,
        ]);
        output.extend_from_slice(&self.global.scoring_cards);
        output.push(u8::from(self.global.habitat_bonuses));

        for (seat, board) in self.boards.iter().enumerate() {
            let player = &self.players[seat];
            if usize::from(player.relative_seat) != seat
                || board.relative_seat != player.relative_seat
            {
                return Err(R4Error::InvalidState(
                    "board/player order changed before packing".to_owned(),
                ));
            }
            write_signed_varint(&mut output, board.center.q);
            write_signed_varint(&mut output, board.center.r);
            output.push(player.nature_tokens);
            output.push(player.occupied_count);
            output.extend_from_slice(&player.wildlife_counts);
            output.extend_from_slice(&player.largest_habitats);
            output.push(board.authority_local_occupied.len() as u8);
            for indexed in &board.authority_local_occupied {
                output.push(indexed.index);
                output.extend_from_slice(&occupied_to_record_entity(&indexed.tile)?[2..]);
            }
            output.push(board.authority_overflow_occupied.len() as u8);
            for tile in &board.authority_overflow_occupied {
                write_signed_varint(&mut output, tile.coord.q);
                write_signed_varint(&mut output, tile.coord.r);
                output.extend_from_slice(&occupied_to_record_entity(tile)?[2..]);
            }
        }

        for token in &self.market {
            output.extend_from_slice(&market_to_record_entity(token));
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
            return Err(R4Error::InvalidPackedMagic);
        }
        let schema = u16::from_le_bytes(cursor.take_array()?);
        if schema != PACKED_SCHEMA_VERSION {
            return Err(R4Error::UnsupportedPackedSchema(schema));
        }
        let flags = u16::from_le_bytes(cursor.take_array()?);
        if flags & !FLAG_SUPPLIED_TILE != 0 {
            return Err(R4Error::UnsupportedPackedFlags(flags));
        }
        let radius_code = cursor.take_u8()?;
        let radius = NearFieldRadius::from_code(radius_code)
            .ok_or(R4Error::InvalidRadiusCode(radius_code))?;
        if cursor.take_u8()? != 0 {
            return Err(R4Error::InvalidState(
                "packed header reserved byte must be zero".to_owned(),
            ));
        }
        let game_index = u64::from_le_bytes(cursor.take_array()?);
        let [turn, active_seat, player_count, total_turns] = cursor.take_array()?;
        if !(1..=BOARD_SLOTS as u8).contains(&player_count) {
            return Err(R4Error::InvalidState(format!(
                "packed player count {player_count} is outside [1, {BOARD_SLOTS}]"
            )));
        }
        let scoring_cards = cursor.take_array()?;
        let habitat_bonuses = match cursor.take_u8()? {
            0 => false,
            1 => true,
            value => {
                return Err(R4Error::InvalidState(format!(
                    "packed habitat-bonus flag {value} is not zero or one"
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
        let mut centers = Vec::with_capacity(usize::from(player_count));
        for relative_seat in 0..player_count {
            let seat = usize::from(relative_seat);
            let center =
                AxialCoord::new(cursor.take_signed_varint()?, cursor.take_signed_varint()?);
            centers.push(center);
            record.nature_tokens[seat] = cursor.take_u8()?;
            let occupied_count = cursor.take_u8()?;
            if usize::from(occupied_count) > MAX_BOARD_TILES {
                return Err(R4Error::InvalidState(format!(
                    "relative seat {relative_seat} occupied count {occupied_count} exceeds {MAX_BOARD_TILES}"
                )));
            }
            record.board_counts[seat] = occupied_count;
            record.wildlife_counts[seat] = cursor.take_array()?;
            record.habitat_sizes[seat] = cursor.take_array()?;

            let local_count = cursor.take_u8()?;
            let mut rows =
                Vec::<[u8; BOARD_ENTITY_SIZE]>::with_capacity(usize::from(occupied_count));
            let mut previous_index = None;
            for _ in 0..local_count {
                let index = cursor.take_u8()?;
                if previous_index.is_some_and(|previous| previous >= index) {
                    return Err(R4Error::InvalidState(
                        "packed local occupied indices are not strictly ordered".to_owned(),
                    ));
                }
                previous_index = Some(index);
                let relative =
                    hex_disk_coord(radius.radius(), u16::from(index)).ok_or_else(|| {
                        R4Error::InvalidState("packed local occupied index is invalid".to_owned())
                    })?;
                let coord = checked_add(relative, center)?;
                let semantic = cursor.take_array::<6>()?;
                rows.push(entity_from_coord_and_semantic(coord, semantic)?);
            }

            let overflow_count = cursor.take_u8()?;
            let mut previous_overflow = None;
            for _ in 0..overflow_count {
                let coord =
                    AxialCoord::new(cursor.take_signed_varint()?, cursor.take_signed_varint()?);
                if previous_overflow.is_some_and(|previous| previous >= coord) {
                    return Err(R4Error::InvalidState(
                        "packed overflow coordinates are not strictly ordered".to_owned(),
                    ));
                }
                previous_overflow = Some(coord);
                let semantic = cursor.take_array::<6>()?;
                rows.push(entity_from_coord_and_semantic(coord, semantic)?);
            }
            if u16::from(local_count) + u16::from(overflow_count) != u16::from(occupied_count) {
                return Err(R4Error::InvalidState(format!(
                    "relative seat {relative_seat} local plus overflow counts do not equal occupied count"
                )));
            }
            rows.sort_unstable_by_key(|row| (row[0] as i8, row[1] as i8));
            if rows.windows(2).any(|pair| pair[0][..2] == pair[1][..2]) {
                return Err(R4Error::InvalidState(
                    "packed occupied coordinates are duplicated".to_owned(),
                ));
            }
            for (row, entity) in rows.into_iter().enumerate() {
                record.board_entities[seat][row] = entity;
            }
        }
        for entity in &mut record.market_entities {
            *entity = cursor.take_array()?;
        }
        let supplied_tile = if flags & FLAG_SUPPLIED_TILE != 0 {
            Some(parse_supplied_tile(cursor.take_array()?)?)
        } else {
            None
        };
        if cursor.remaining() != 0 {
            return Err(R4Error::TrailingPackedBytes(cursor.remaining()));
        }

        let sparse = SparsePublicState::from_position_record(&record, supplied_tile)?;
        let state = AdaptiveMultiResolutionState::assemble(&sparse, radius, &centers)?;
        if state.to_packed_bytes()? != bytes {
            return Err(R4Error::InvalidState(
                "decode followed by encode changed the packed byte stream".to_owned(),
            ));
        }
        Ok(state)
    }
}

fn parse_supplied_tile(bytes: [u8; 4]) -> Result<SuppliedTile> {
    let terrain_a = terrain_from_code(bytes[0]).ok_or_else(|| {
        R4Error::InvalidState("supplied-tile primary terrain code is invalid".to_owned())
    })?;
    let terrain_b = if bytes[1] == NONE {
        None
    } else {
        Some(terrain_from_code(bytes[1]).ok_or_else(|| {
            R4Error::InvalidState("supplied-tile secondary terrain code is invalid".to_owned())
        })?)
    };
    let wildlife_eligibility = cascadia_game::WildlifeMask::from_bits(bytes[2]);
    if wildlife_eligibility.bits() != bytes[2] {
        return Err(R4Error::InvalidState(
            "supplied-tile wildlife mask is invalid".to_owned(),
        ));
    }
    let keystone = match bytes[3] {
        0 => false,
        1 => true,
        value => {
            return Err(R4Error::InvalidState(format!(
                "supplied-tile keystone flag {value} is invalid"
            )));
        }
    };
    let tile = SuppliedTile {
        terrain_a,
        terrain_b,
        wildlife_eligibility,
        keystone,
    };
    tile.validate()?;
    Ok(tile)
}

const fn terrain_from_code(code: u8) -> Option<cascadia_game::Terrain> {
    match code {
        0 => Some(cascadia_game::Terrain::Mountain),
        1 => Some(cascadia_game::Terrain::Forest),
        2 => Some(cascadia_game::Terrain::Prairie),
        3 => Some(cascadia_game::Terrain::Wetland),
        4 => Some(cascadia_game::Terrain::River),
        _ => None,
    }
}

fn entity_from_coord_and_semantic(
    coord: AxialCoord,
    semantic: [u8; 6],
) -> Result<[u8; BOARD_ENTITY_SIZE]> {
    let q = i8::try_from(coord.q)
        .map_err(|_| R4Error::InvalidState("packed q coordinate is out of range".to_owned()))?;
    let r = i8::try_from(coord.r)
        .map_err(|_| R4Error::InvalidState("packed r coordinate is out of range".to_owned()))?;
    Ok([
        q as u8,
        r as u8,
        semantic[0],
        semantic[1],
        semantic[2],
        semantic[3],
        semantic[4],
        semantic[5],
    ])
}

fn checked_add(relative: AxialCoord, center: AxialCoord) -> Result<AxialCoord> {
    Ok(AxialCoord::new(
        relative
            .q
            .checked_add(center.q)
            .ok_or_else(|| R4Error::InvalidState("packed q addition overflowed".to_owned()))?,
        relative
            .r
            .checked_add(center.r)
            .ok_or_else(|| R4Error::InvalidState("packed r addition overflowed".to_owned()))?,
    ))
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
        Ok(self.take_array::<1>()?[0])
    }

    fn take_array<const N: usize>(&mut self) -> Result<[u8; N]> {
        let end = self
            .offset
            .checked_add(N)
            .ok_or(R4Error::UnexpectedPackedEnd)?;
        let value = self
            .bytes
            .get(self.offset..end)
            .ok_or(R4Error::UnexpectedPackedEnd)?
            .try_into()
            .expect("packed slice length was checked");
        self.offset = end;
        Ok(value)
    }

    fn take_signed_varint(&mut self) -> Result<i16> {
        let start = self.offset;
        let mut value = 0u32;
        let mut shift = 0;
        loop {
            if shift >= 21 {
                return Err(R4Error::VarintOverflow);
            }
            let byte = self.take_u8()?;
            value |= u32::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                break;
            }
            shift += 7;
        }
        if value > u32::from(u16::MAX) {
            return Err(R4Error::VarintOverflow);
        }
        let decoded = ((value >> 1) as i32) ^ -((value & 1) as i32);
        let decoded = i16::try_from(decoded).map_err(|_| R4Error::VarintOverflow)?;
        let mut canonical = Vec::new();
        write_signed_varint(&mut canonical, decoded);
        if self.bytes[start..self.offset] != canonical {
            return Err(R4Error::NonCanonicalVarint);
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
        fn signed_varint_round_trips(value in any::<i16>()) {
            let mut bytes = Vec::new();
            write_signed_varint(&mut bytes, value);
            let mut cursor = ByteCursor::new(&bytes);
            prop_assert_eq!(cursor.take_signed_varint().unwrap(), value);
            prop_assert_eq!(cursor.remaining(), 0);
        }
    }

    #[test]
    fn signed_varint_rejects_overlong_bytes() {
        let mut cursor = ByteCursor::new(&[0x80, 0x00]);
        assert!(matches!(
            cursor.take_signed_varint(),
            Err(R4Error::NonCanonicalVarint)
        ));
    }
}
