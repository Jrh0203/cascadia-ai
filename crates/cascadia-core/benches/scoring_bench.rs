use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::SeedableRng;
use rand::rngs::StdRng;

use cascadia_core::board::Board;
use cascadia_core::hex::HexCoord;
use cascadia_core::scoring::ScoreBreakdown;
use cascadia_core::types::*;

/// Build a realistic mid-game board with ~15 tiles and ~10 wildlife tokens.
fn build_test_board() -> Board {
    let mut board = Board::new();

    let tiles = [
        (0, 0, Terrain::Forest, Some(Terrain::River), &[Wildlife::Bear, Wildlife::Salmon][..]),
        (1, 0, Terrain::Forest, None, &[Wildlife::Bear][..]),
        (0, 1, Terrain::Prairie, Some(Terrain::Forest), &[Wildlife::Elk, Wildlife::Fox][..]),
        (-1, 0, Terrain::Mountain, Some(Terrain::Wetland), &[Wildlife::Hawk, Wildlife::Elk][..]),
        (0, -1, Terrain::River, Some(Terrain::Wetland), &[Wildlife::Salmon][..]),
        (1, -1, Terrain::Prairie, Some(Terrain::Mountain), &[Wildlife::Fox, Wildlife::Bear][..]),
        (-1, 1, Terrain::Wetland, None, &[Wildlife::Salmon][..]),
        (2, 0, Terrain::Forest, Some(Terrain::Prairie), &[Wildlife::Elk, Wildlife::Hawk][..]),
        (1, 1, Terrain::Mountain, Some(Terrain::River), &[Wildlife::Bear, Wildlife::Hawk][..]),
        (-1, -1, Terrain::Prairie, Some(Terrain::Forest), &[Wildlife::Fox][..]),
        (2, -1, Terrain::Forest, Some(Terrain::Mountain), &[Wildlife::Elk][..]),
        (-2, 1, Terrain::River, Some(Terrain::Wetland), &[Wildlife::Salmon, Wildlife::Hawk][..]),
        (0, 2, Terrain::Wetland, Some(Terrain::Prairie), &[Wildlife::Fox, Wildlife::Bear][..]),
        (-1, 2, Terrain::Mountain, None, &[Wildlife::Hawk][..]),
        (2, 1, Terrain::River, Some(Terrain::Forest), &[Wildlife::Salmon, Wildlife::Elk][..]),
    ];

    for &(q, r, t1, t2, allowed) in &tiles {
        let mask = WildlifeMask::new(allowed);
        let tile = match t2 {
            Some(t2) => TileData::dual(t1, t2, mask),
            None => TileData::single(t1, mask),
        };
        board.place_tile(HexCoord::new(q, r), tile, 0);
    }

    // Place wildlife tokens
    let wildlife_placements = [
        (0, 0, Wildlife::Bear),
        (1, 0, Wildlife::Bear),
        (0, 1, Wildlife::Elk),
        (-1, 0, Wildlife::Hawk),
        (0, -1, Wildlife::Salmon),
        (1, -1, Wildlife::Fox),
        (-1, 1, Wildlife::Salmon),
        (2, 0, Wildlife::Elk),
        (-1, -1, Wildlife::Fox),
        (2, -1, Wildlife::Elk),
    ];

    for &(q, r, w) in &wildlife_placements {
        let idx = HexCoord::new(q, r).to_index().unwrap();
        board.place_wildlife(idx, w);
    }

    board
}

fn bench_full_scoring(c: &mut Criterion) {
    let mut board = build_test_board();
    let cards = ScoringCards::all_a();

    c.bench_function("full_board_score", |b| {
        b.iter(|| {
            black_box(ScoreBreakdown::compute(black_box(&mut board), black_box(&cards)))
        })
    });
}

fn bench_wildlife_scoring(c: &mut Criterion) {
    let board = build_test_board();
    let cards = ScoringCards::all_a();

    c.bench_function("wildlife_score_all", |b| {
        b.iter(|| {
            black_box(cascadia_core::scoring::wildlife::score_all_wildlife(
                black_box(&board),
                black_box(&cards),
            ))
        })
    });
}

fn bench_tile_placement(c: &mut Criterion) {
    let base_board = build_test_board();
    let frontier = base_board.frontier();
    let tile = TileData::dual(
        Terrain::Forest,
        Terrain::River,
        WildlifeMask::new(&[Wildlife::Salmon]),
    );

    c.bench_function("tile_place_and_undo", |b| {
        let mut board = base_board.clone();
        let coord = HexCoord::from_index(frontier[0] as usize);
        b.iter(|| {
            let action = board.place_tile(coord, tile, 0).unwrap();
            black_box(&board.largest_group);
            board.undo(action);
        })
    });
}

criterion_group!(benches, bench_full_scoring, bench_wildlife_scoring, bench_tile_placement);
criterion_main!(benches);
