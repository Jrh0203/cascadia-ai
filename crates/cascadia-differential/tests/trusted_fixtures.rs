use cascadia_core as v1;
use cascadia_game as v2;

#[derive(Clone, Copy)]
struct WildlifePlacement {
    q: i8,
    r: i8,
    wildlife: Option<WildlifeKind>,
}

#[derive(Clone, Copy)]
enum WildlifeKind {
    Bear,
    Elk,
    Salmon,
    Hawk,
    Fox,
}

const AAAAA_WILDLIFE_FIXTURES: &[(&str, &[WildlifePlacement], [u16; 5])] = &[
    (
        "two isolated bear pairs",
        &[
            placement(0, 0, WildlifeKind::Bear),
            placement(1, 0, WildlifeKind::Bear),
            filler(2, 0),
            filler(3, 0),
            placement(4, 0, WildlifeKind::Bear),
            placement(5, 0, WildlifeKind::Bear),
        ],
        [11, 0, 0, 0, 0],
    ),
    (
        "four elk in a straight line",
        &[
            placement(0, 0, WildlifeKind::Elk),
            placement(1, 0, WildlifeKind::Elk),
            placement(2, 0, WildlifeKind::Elk),
            placement(3, 0, WildlifeKind::Elk),
        ],
        [0, 13, 0, 0, 0],
    ),
    (
        "four salmon in a nonbranching run",
        &[
            placement(0, 0, WildlifeKind::Salmon),
            placement(1, 0, WildlifeKind::Salmon),
            placement(2, 0, WildlifeKind::Salmon),
            placement(3, 0, WildlifeKind::Salmon),
        ],
        [0, 0, 12, 0, 0],
    ),
    (
        "two nonadjacent hawks",
        &[
            placement(0, 0, WildlifeKind::Hawk),
            filler(1, 0),
            filler(2, 0),
            placement(3, 0, WildlifeKind::Hawk),
        ],
        [0, 0, 0, 5, 0],
    ),
    (
        "fox adjacent to two wildlife species",
        &[
            placement(0, 0, WildlifeKind::Fox),
            placement(1, 0, WildlifeKind::Bear),
            placement(0, 1, WildlifeKind::Elk),
        ],
        [0, 2, 0, 0, 2],
    ),
];

const fn placement(q: i8, r: i8, wildlife: WildlifeKind) -> WildlifePlacement {
    WildlifePlacement {
        q,
        r,
        wildlife: Some(wildlife),
    }
}

const fn filler(q: i8, r: i8) -> WildlifePlacement {
    WildlifePlacement {
        q,
        r,
        wildlife: None,
    }
}

#[test]
fn independently_verified_aaaaa_wildlife_fixtures_match_both_engines() {
    for (name, placements, expected) in AAAAA_WILDLIFE_FIXTURES {
        let v1_score = score_v1_wildlife(placements);
        let v2_score = score_v2_wildlife(placements);
        assert_eq!(v1_score, *expected, "v1 fixture: {name}");
        assert_eq!(v2_score, *expected, "v2 fixture: {name}");
        assert_eq!(v2_score, v1_score, "differential fixture: {name}");
    }
}

#[test]
fn connected_elk_lines_use_the_highest_scoring_partition() {
    // Official rules require connected elk to use the interpretation with the
    // largest score. The vertical line of three scores 9 and leaves an
    // adjacent vertical line of two scoring 5, for 14 total.
    let placements = [
        placement(0, 1, WildlifeKind::Elk),
        placement(0, 2, WildlifeKind::Elk),
        placement(-1, 1, WildlifeKind::Elk),
        placement(0, 3, WildlifeKind::Elk),
        placement(-1, 2, WildlifeKind::Elk),
    ];

    assert_eq!(score_v2_wildlife(&placements)[1], 14);
    assert_eq!(
        score_v1_wildlife(&placements)[1],
        13,
        "the historical v1 greedy partition is intentionally captured as a known rules mismatch"
    );
}

#[test]
fn four_tile_forest_chain_matches_both_engines() {
    let mut v1_board = v1::board::Board::new();
    let mut v2_board = v2::Board::empty();
    for q in 0i8..4 {
        v1_board
            .place_tile(
                v1::hex::HexCoord::new(q, 0),
                v1::types::TileData::single(
                    v1::types::Terrain::Forest,
                    v1::types::WildlifeMask::new(&[v1::types::Wildlife::Bear]),
                ),
                0,
            )
            .expect("trusted v1 habitat placement is legal");
        v2_board
            .place_tile(
                v2::HexCoord::new(q, 0),
                v2::Tile::keystone(q as u8, v2::Terrain::Forest, v2::Wildlife::Bear),
                v2::Rotation::ZERO,
            )
            .expect("trusted v2 habitat placement is legal");
    }

    let v1_score =
        v1::scoring::ScoreBreakdown::compute(&mut v1_board, &v1::types::ScoringCards::all_a());
    let v2_score = v2::score_board(&v2_board, v2::ScoringCards::AAAAA);
    let v1_habitats = [
        v1_score.habitat[v1::types::Terrain::Mountain as usize],
        v1_score.habitat[v1::types::Terrain::Forest as usize],
        v1_score.habitat[v1::types::Terrain::Prairie as usize],
        v1_score.habitat[v1::types::Terrain::Wetland as usize],
        v1_score.habitat[v1::types::Terrain::River as usize],
    ];

    assert_eq!(v1_habitats, [0, 4, 0, 0, 0]);
    assert_eq!(v2_score.habitat, [0, 4, 0, 0, 0]);
    assert_eq!(v2_score.base_total, 4);
    assert_eq!(v2_score.base_total, v1_score.total);
}

fn score_v1_wildlife(placements: &[WildlifePlacement]) -> [u16; 5] {
    let mut board = v1::board::Board::new();
    for placement in placements {
        let coord = v1::hex::HexCoord::new(placement.q, placement.r);
        board
            .place_tile(
                coord,
                v1::types::TileData::dual(
                    v1::types::Terrain::Forest,
                    v1::types::Terrain::Prairie,
                    v1::types::WildlifeMask::new(&v1::types::Wildlife::ALL),
                ),
                0,
            )
            .expect("trusted v1 wildlife fixture tile is legal");
        if let Some(wildlife) = placement.wildlife {
            board
                .place_wildlife(
                    coord.to_index().expect("trusted coordinate is in bounds"),
                    v1_wildlife(wildlife),
                )
                .expect("trusted v1 wildlife placement is legal");
        }
    }
    v1::scoring::ScoreBreakdown::compute(&mut board, &v1::types::ScoringCards::all_a()).wildlife
}

fn score_v2_wildlife(placements: &[WildlifePlacement]) -> [u16; 5] {
    let mut board = v2::Board::empty();
    for (index, placement) in placements.iter().enumerate() {
        let coord = v2::HexCoord::new(placement.q, placement.r);
        board
            .place_tile(
                coord,
                v2::Tile::dual(
                    index as u8,
                    v2::Terrain::Forest,
                    v2::Terrain::Prairie,
                    0b1_1111,
                ),
                v2::Rotation::ZERO,
            )
            .expect("trusted v2 wildlife fixture tile is legal");
        if let Some(wildlife) = placement.wildlife {
            board
                .place_wildlife(coord, v2_wildlife(wildlife))
                .expect("trusted v2 wildlife placement is legal");
        }
    }
    v2::score_board(&board, v2::ScoringCards::AAAAA).wildlife
}

const fn v1_wildlife(wildlife: WildlifeKind) -> v1::types::Wildlife {
    match wildlife {
        WildlifeKind::Bear => v1::types::Wildlife::Bear,
        WildlifeKind::Elk => v1::types::Wildlife::Elk,
        WildlifeKind::Salmon => v1::types::Wildlife::Salmon,
        WildlifeKind::Hawk => v1::types::Wildlife::Hawk,
        WildlifeKind::Fox => v1::types::Wildlife::Fox,
    }
}

const fn v2_wildlife(wildlife: WildlifeKind) -> v2::Wildlife {
    match wildlife {
        WildlifeKind::Bear => v2::Wildlife::Bear,
        WildlifeKind::Elk => v2::Wildlife::Elk,
        WildlifeKind::Salmon => v2::Wildlife::Salmon,
        WildlifeKind::Hawk => v2::Wildlife::Hawk,
        WildlifeKind::Fox => v2::Wildlife::Fox,
    }
}
