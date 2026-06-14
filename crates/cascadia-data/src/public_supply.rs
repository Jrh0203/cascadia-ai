use cascadia_game::PublicSupply;

pub const PUBLIC_SUPPLY_SIZE: usize = 30;

pub(crate) fn encode_public_supply(supply: PublicSupply) -> [u8; PUBLIC_SUPPLY_SIZE] {
    let mut bytes = [0; PUBLIC_SUPPLY_SIZE];
    let mut offset = 0;
    for values in [
        supply.wildlife_bag.as_slice(),
        supply.unseen_tile_terrain_capacity.as_slice(),
        supply.unseen_tile_wildlife_capacity.as_slice(),
        supply.unseen_keystones_by_terrain.as_slice(),
        supply.unseen_dual_terrain_pairs.as_slice(),
    ] {
        bytes[offset..offset + values.len()].copy_from_slice(values);
        offset += values.len();
    }
    debug_assert_eq!(offset, PUBLIC_SUPPLY_SIZE);
    bytes
}

pub(crate) fn decode_public_supply(bytes: [u8; PUBLIC_SUPPLY_SIZE]) -> PublicSupply {
    let mut offset = 0;
    let mut take = |length: usize| {
        let start = offset;
        offset += length;
        &bytes[start..offset]
    };
    let wildlife_bag = take(5).try_into().expect("fixed public supply");
    let unseen_tile_terrain_capacity = take(5).try_into().expect("fixed public supply");
    let unseen_tile_wildlife_capacity = take(5).try_into().expect("fixed public supply");
    let unseen_keystones_by_terrain = take(5).try_into().expect("fixed public supply");
    let unseen_dual_terrain_pairs = take(10).try_into().expect("fixed public supply");
    debug_assert_eq!(offset, PUBLIC_SUPPLY_SIZE);
    PublicSupply {
        wildlife_bag,
        unseen_tile_terrain_capacity,
        unseen_tile_wildlife_capacity,
        unseen_keystones_by_terrain,
        unseen_dual_terrain_pairs,
    }
}

#[cfg(test)]
mod tests {
    use cascadia_game::{GameConfig, GameSeed, GameState};

    use super::*;

    #[test]
    fn public_supply_encoding_round_trips_exactly() {
        let game = GameState::new(
            GameConfig::research_aaaaa(4).unwrap(),
            GameSeed::from_u64(91),
        )
        .unwrap();
        let expected = game.public_supply();
        assert_eq!(
            decode_public_supply(encode_public_supply(expected)),
            expected
        );
    }
}
