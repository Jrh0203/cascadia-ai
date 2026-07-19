//! Integration tests proving the v3 alternate scoring-card implementations
//! match the legacy engine-verified semantics for the CBDDB research set:
//! Bear C, Elk B, Salmon D, Hawk D, Fox B.
//!
//! Reference implementations and test suites live on the archive branch
//! `archive/pre-v3-repo-cleanup-2026-07-01` under
//! `legacy/crates/cascadia-core/src/scoring/wildlife/{bear,elk,salmon,hawk,fox,matching}.rs`.
//! Every legacy unit test that exercises one of the five cards is ported here,
//! together with table-driven cases covering every threshold breakpoint of
//! each card's official scoring table.
//!
//! The v3 per-animal scorers (`score_bears`, `score_elk`, ...) are private, so
//! all assertions go through the public `score_board` entry point with
//! `ScoringCards::CBDDB` and read the target animal's slot of
//! `ScoreBreakdown::wildlife`. Boards are built through the public
//! `place_tile`/`place_wildlife` API, which enforces tile connectivity; a
//! connector spine of empty tiles (no wildlife placed) keeps every fixture
//! connected. Empty tiles are invisible to all five wildlife scorers, so the
//! spine never affects the asserted scores.
//!
//! Legacy `matching.rs` unit tests call `max_weight_matching` directly, which
//! is `pub(crate)` in both engines. They are ported as board-level Hawk D
//! equivalents that force the bitmask matching DP through the same structural
//! decisions (single edge, node-sharing triangle, disjoint pairs, chain
//! partition, three simultaneous pairs).

use cascadia_game::{
    Board, HexCoord, MAX_BOARD_TILES, Rotation, ScoringCards, Terrain, Tile, TileId, Wildlife,
    WildlifeMask, score_board,
};

/// A non-keystone tile that accepts every wildlife type, mirroring the legacy
/// test fixtures (`TileData::single/dual(..., WildlifeMask::new(&Wildlife::ALL))`)
/// and the v3 unit-test fixture in `scoring.rs`.
fn fixture_tile(id: u8) -> Tile {
    Tile {
        id: TileId(id),
        terrain_a: Terrain::Forest,
        terrain_b: None,
        wildlife: WildlifeMask::from_bits(0b1_1111),
        keystone: false,
    }
}

fn push_unique(coords: &mut Vec<HexCoord>, coord: HexCoord) {
    if !coords.contains(&coord) {
        coords.push(coord);
    }
}

/// Builds a board holding exactly the given wildlife tokens. The public
/// placement API requires every tile to touch the existing board, so each
/// entry is connected to the origin along an axis-aligned path of empty
/// connector tiles (q-steps `(±1, 0)` then r-steps `(0, ±1)`, both hex
/// directions). Connector tiles carry no wildlife and therefore cannot change
/// any wildlife score.
fn build_board(entries: &[(i8, i8, Wildlife)]) -> Board {
    let mut coords = vec![HexCoord::new(0, 0)];
    for &(q, r, _) in entries {
        let mut cur = HexCoord::new(0, 0);
        while cur.q != q {
            cur = HexCoord::new(cur.q + (q - cur.q).signum(), cur.r);
            push_unique(&mut coords, cur);
        }
        while cur.r != r {
            cur = HexCoord::new(cur.q, cur.r + (r - cur.r).signum());
            push_unique(&mut coords, cur);
        }
    }
    assert!(
        coords.len() <= MAX_BOARD_TILES,
        "test board needs {} tiles, exceeding the {} tile cap",
        coords.len(),
        MAX_BOARD_TILES
    );

    // Order cells so that every placement after the first touches an already
    // placed tile. The connector paths make the set connected, so this loop
    // always terminates.
    let mut ordered = vec![coords[0]];
    while ordered.len() < coords.len() {
        let before = ordered.len();
        for &coord in &coords {
            if ordered.contains(&coord) {
                continue;
            }
            if coord
                .neighbors()
                .into_iter()
                .any(|neighbor| ordered.contains(&neighbor))
            {
                ordered.push(coord);
            }
        }
        assert!(before < ordered.len(), "connector spine is not connected");
    }

    let mut board = Board::empty();
    for (index, coord) in ordered.iter().enumerate() {
        board
            .place_tile(*coord, fixture_tile(index as u8 + 1), Rotation::ZERO)
            .expect("fixture tile placement is legal");
        if let Some(entry) = entries
            .iter()
            .find(|entry| HexCoord::new(entry.0, entry.1) == *coord)
        {
            board
                .place_wildlife(*coord, entry.2)
                .expect("fixture wildlife placement is legal");
        }
    }
    board
}

fn wildlife_score(entries: &[(i8, i8, Wildlife)], animal: Wildlife) -> u16 {
    score_board(&build_board(entries), ScoringCards::CBDDB).wildlife[animal as usize]
}

use Wildlife::{Bear, Elk, Fox, Hawk, Salmon};

// =====================================================================
// Shared edge case: empty board scores zero on all five cards.
// Ports legacy `no_bears` / `no_elk` / `no_salmon` / `no_hawks` / `no_foxes`.
// =====================================================================

#[test]
fn empty_board_scores_zero_on_all_cbddb_cards() {
    let breakdown = score_board(&Board::empty(), ScoringCards::CBDDB);
    assert_eq!(breakdown.wildlife, [0, 0, 0, 0, 0]);
    assert_eq!(breakdown.total, 0);
}

// =====================================================================
// Bear C: group of 1 = 2, group of 2 = 5, group of 3 = 8, 4+ = 0.
// Bonus +3 when at least one group of each size 1, 2, and 3 exists.
// =====================================================================

#[test]
fn bear_c_single_bear_scores_two() {
    assert_eq!(wildlife_score(&[(0, 0, Bear)], Bear), 2);
}

/// Ports legacy `c_singletons_only` (compressed from gap 5 to gap 3; only
/// component separation matters).
#[test]
fn bear_c_two_singletons_score_four() {
    assert_eq!(wildlife_score(&[(0, 0, Bear), (3, 0, Bear)], Bear), 4);
}

#[test]
fn bear_c_pair_scores_five() {
    assert_eq!(wildlife_score(&[(0, 0, Bear), (1, 0, Bear)], Bear), 5);
}

#[test]
fn bear_c_triangle_scores_eight() {
    assert_eq!(
        wildlife_score(&[(0, 0, Bear), (1, 0, Bear), (0, 1, Bear)], Bear),
        8
    );
}

/// Ports legacy `c_group_of_four_doesnt_score`.
#[test]
fn bear_c_group_of_four_scores_zero() {
    assert_eq!(
        wildlife_score(
            &[(0, 0, Bear), (1, 0, Bear), (0, 1, Bear), (1, 1, Bear)],
            Bear
        ),
        0
    );
}

#[test]
fn bear_c_group_of_five_scores_zero() {
    assert_eq!(
        wildlife_score(
            &[
                (0, 0, Bear),
                (1, 0, Bear),
                (0, 1, Bear),
                (1, 1, Bear),
                (2, 1, Bear),
            ],
            Bear
        ),
        0
    );
}

/// Ports legacy `c_one_of_each_with_bonus`: singleton + pair + triangle
/// = 2 + 5 + 8 + 3 bonus = 18.
#[test]
fn bear_c_one_of_each_size_earns_bonus() {
    assert_eq!(
        wildlife_score(
            &[
                (0, 0, Bear),
                // pair, separated from the singleton by two empty tiles
                (3, 0, Bear),
                (4, 0, Bear),
                // triangle, separated from both other groups
                (0, 3, Bear),
                (0, 4, Bear),
                (1, 3, Bear),
            ],
            Bear
        ),
        18
    );
}

/// Ports legacy `c_no_bonus_when_one_size_missing`: singleton + triangle,
/// no pair = 2 + 8, no +3.
#[test]
fn bear_c_no_bonus_when_one_size_missing() {
    assert_eq!(
        wildlife_score(
            &[(0, 0, Bear), (0, 3, Bear), (0, 4, Bear), (1, 3, Bear)],
            Bear
        ),
        10
    );
}

/// Oversize groups are ignored while other groups still score: pair (5) next
/// to a dead 4-group (0) = 5.
#[test]
fn bear_c_oversize_group_ignored_but_pair_still_scores() {
    assert_eq!(
        wildlife_score(
            &[
                (0, 0, Bear),
                (1, 0, Bear),
                (0, 3, Bear),
                (1, 3, Bear),
                (0, 4, Bear),
                (1, 4, Bear),
            ],
            Bear
        ),
        5
    );
}

// =====================================================================
// Elk B: partition elk into shapes — single = 2, adjacent pair = 5,
// triangle = 9, triangle + attached 4th = 13; maximize the total.
// =====================================================================

/// Ports legacy `b_single_elk`.
#[test]
fn elk_b_single_scores_two() {
    assert_eq!(wildlife_score(&[(0, 0, Elk)], Elk), 2);
}

/// Ports legacy `b_pair`.
#[test]
fn elk_b_pair_scores_five() {
    assert_eq!(wildlife_score(&[(0, 0, Elk), (1, 0, Elk)], Elk), 5);
}

/// Ports legacy `b_triangle_scores_nine`.
#[test]
fn elk_b_triangle_scores_nine() {
    assert_eq!(
        wildlife_score(&[(0, 0, Elk), (1, 0, Elk), (0, 1, Elk)], Elk),
        9
    );
}

/// Ports legacy `b_line_of_three_partitions`: pair + single = 7.
#[test]
fn elk_b_line_of_three_partitions_to_seven() {
    assert_eq!(
        wildlife_score(&[(0, 0, Elk), (1, 0, Elk), (2, 0, Elk)], Elk),
        7
    );
}

/// Ports legacy `b_rhombus_scores_thirteen`: triangle plus a 4th elk adjacent
/// to two triangle members.
#[test]
fn elk_b_rhombus_scores_thirteen() {
    assert_eq!(
        wildlife_score(&[(0, 0, Elk), (1, 0, Elk), (0, 1, Elk), (1, 1, Elk)], Elk),
        13
    );
}

/// Ports legacy `b_line_of_four_partitions`: two pairs = 10.
#[test]
fn elk_b_line_of_four_partitions_to_ten() {
    assert_eq!(
        wildlife_score(&[(0, 0, Elk), (1, 0, Elk), (2, 0, Elk), (3, 0, Elk)], Elk),
        10
    );
}

/// Derived from the legacy `score_b_component` semantics (Option D): the 4th
/// elk of a 13-point shape only has to be hex-adjacent to AT LEAST ONE member
/// of the triangle (`adj[i] | adj[j] | adj[k]`). Here the tail elk (2,0) is
/// adjacent solely to triangle member (1,0), so legacy scores the component
/// as one 4-shape = 13. The v3 scorer requires adjacency to at least two
/// triangle members and partitions this as triangle + single = 11.
/// RULED 2026-07-19 (John): the strict-diamond semantics stand for the
/// CBDDB identity — the 4-elk shape must be a true rhombus (4th elk
/// adjacent to two triangle members), matching the official card. The
/// legacy engine scored this board 13 via a looser any-member rule;
/// that divergence is historical only, so the April-2026 alt-rules
/// anchors (~96.5/97.2) are slightly generous relative to this rule.
#[test]
fn elk_b_triangle_with_tail_partitions_as_triangle_plus_single() {
    assert_eq!(
        wildlife_score(&[(0, 0, Elk), (1, 0, Elk), (0, 1, Elk), (2, 0, Elk)], Elk),
        11
    );
}

/// Disconnected components score independently: pair + triangle = 5 + 9.
#[test]
fn elk_b_disjoint_pair_and_triangle_score_fourteen() {
    assert_eq!(
        wildlife_score(
            &[
                (0, 0, Elk),
                (1, 0, Elk),
                (0, 3, Elk),
                (0, 4, Elk),
                (1, 3, Elk),
            ],
            Elk
        ),
        14
    );
}

/// Five connected elk: best partition is rhombus (13) + single (2) = 15.
/// Both engines agree here because the rhombus 4th member touches two
/// triangle members.
#[test]
fn elk_b_rhombus_plus_attached_fifth_scores_fifteen() {
    assert_eq!(
        wildlife_score(
            &[
                (0, 0, Elk),
                (1, 0, Elk),
                (0, 1, Elk),
                (1, 1, Elk),
                (2, 1, Elk),
            ],
            Elk
        ),
        15
    );
}

// =====================================================================
// Salmon D: valid runs (every salmon has ≤ 2 salmon neighbors) of length
// ≥ 3 score 1 point per salmon plus 1 point per unique adjacent CELL holding
// a non-salmon wildlife token. Runs of length 1-2 and branching components
// score 0.
// =====================================================================

/// Ports legacy `d_run_with_no_adjacent_animals` plus derived length
/// breakpoints (1 and 2 below the minimum, 3-5 scoring their length).
#[test]
fn salmon_d_run_length_table_without_neighbors() {
    for (len, expected) in [(1i8, 0u16), (2, 0), (3, 3), (4, 4), (5, 5)] {
        let entries: Vec<(i8, i8, Wildlife)> = (0..len).map(|q| (q, 0, Salmon)).collect();
        assert_eq!(
            wildlife_score(&entries, Salmon),
            expected,
            "run length {len}"
        );
    }
}

/// Ports legacy `d_short_runs_dont_score`: length 2 with an adjacent bear
/// still scores 0.
#[test]
fn salmon_d_short_run_with_animals_scores_zero() {
    assert_eq!(
        wildlife_score(&[(0, 0, Salmon), (1, 0, Salmon), (0, 1, Bear)], Salmon),
        0
    );
}

/// Ports legacy `d_single_salmon_doesnt_score`.
#[test]
fn salmon_d_single_salmon_with_animals_scores_zero() {
    assert_eq!(
        wildlife_score(&[(0, 0, Salmon), (1, 0, Bear), (-1, 0, Elk)], Salmon),
        0
    );
}

/// Ports legacy `d_run_with_adjacent_animals`: 3 salmon + bear and elk at the
/// ends = 5.
#[test]
fn salmon_d_run_with_two_adjacent_animals_scores_five() {
    assert_eq!(
        wildlife_score(
            &[
                (0, 0, Salmon),
                (1, 0, Salmon),
                (2, 0, Salmon),
                (-1, 0, Bear),
                (3, 0, Elk),
            ],
            Salmon
        ),
        5
    );
}

/// Ports legacy `d_animal_adjacent_to_two_salmon_counted_once`: the bear at
/// (1,-1) borders both (0,0) and (1,0) but adds only 1 point.
#[test]
fn salmon_d_token_adjacent_to_two_salmon_counts_once() {
    assert_eq!(
        wildlife_score(
            &[
                (0, 0, Salmon),
                (1, 0, Salmon),
                (2, 0, Salmon),
                (1, -1, Bear),
            ],
            Salmon
        ),
        4
    );
}

/// Adjacent tokens are counted per CELL, not per type: two bears in different
/// cells add 2 points.
#[test]
fn salmon_d_two_tokens_of_same_type_count_per_cell() {
    assert_eq!(
        wildlife_score(
            &[
                (0, 0, Salmon),
                (1, 0, Salmon),
                (2, 0, Salmon),
                (-1, 0, Bear),
                (3, 0, Bear),
            ],
            Salmon
        ),
        5
    );
}

/// Ports legacy `d_branching_scores_zero`: a Y-shaped component scores 0 even
/// with animals on the board.
#[test]
fn salmon_d_branching_scores_zero_even_with_animals() {
    assert_eq!(
        wildlife_score(
            &[
                (0, 0, Salmon),
                (1, 0, Salmon),
                (-1, 0, Salmon),
                (0, -1, Salmon),
                (3, 0, Bear),
            ],
            Salmon
        ),
        0
    );
}

/// A closed 6-ring is a valid run (every salmon has exactly 2 salmon
/// neighbors) and the enclosed token counts once: 6 + 1 = 7.
#[test]
fn salmon_d_ring_run_counts_interior_token_once() {
    assert_eq!(
        wildlife_score(
            &[
                (0, 0, Elk),
                (1, 0, Salmon),
                (1, -1, Salmon),
                (0, -1, Salmon),
                (-1, 0, Salmon),
                (-1, 1, Salmon),
                (0, 1, Salmon),
            ],
            Salmon
        ),
        7
    );
}

/// Two separate runs score independently: (3 salmon + 1 bear) + 3 salmon = 7.
#[test]
fn salmon_d_two_runs_score_independently() {
    assert_eq!(
        wildlife_score(
            &[
                (0, 0, Salmon),
                (1, 0, Salmon),
                (2, 0, Salmon),
                (-1, 0, Bear),
                (0, 3, Salmon),
                (1, 3, Salmon),
                (2, 3, Salmon),
            ],
            Salmon
        ),
        7
    );
}

// =====================================================================
// Hawk D: pairs of non-adjacent hawks in line of sight score by the number
// of unique non-hawk wildlife TYPES strictly between them: 1 = 4, 2 = 7,
// 3+ = 9 (0 types = no pair). Each hawk joins at most one pair; the pairing
// maximizes the total (bitmask matching DP).
// =====================================================================

/// Ports legacy `d_no_pairs`: hawks off every shared hex axis never pair.
#[test]
fn hawk_d_off_axis_pair_scores_zero() {
    assert_eq!(wildlife_score(&[(0, 0, Hawk), (2, 1, Hawk)], Hawk), 0);
}

#[test]
fn hawk_d_adjacent_hawks_score_zero() {
    assert_eq!(wildlife_score(&[(0, 0, Hawk), (1, 0, Hawk)], Hawk), 0);
}

/// A line-of-sight pair with no wildlife between scores 0 (0 types).
#[test]
fn hawk_d_pair_with_empty_between_scores_zero() {
    assert_eq!(wildlife_score(&[(0, 0, Hawk), (2, 0, Hawk)], Hawk), 0);
}

/// Ports legacy `d_pair_with_one_animal`: one type between = 4.
#[test]
fn hawk_d_pair_with_one_type_scores_four() {
    assert_eq!(
        wildlife_score(&[(0, 0, Hawk), (3, 0, Hawk), (1, 0, Bear)], Hawk),
        4
    );
}

/// Ports legacy `d_pair_with_two_types`: bear + elk + duplicate bear between
/// = 2 unique types = 7.
#[test]
fn hawk_d_pair_with_two_types_dedupes_to_seven() {
    assert_eq!(
        wildlife_score(
            &[
                (0, 0, Hawk),
                (4, 0, Hawk),
                (1, 0, Bear),
                (2, 0, Elk),
                (3, 0, Bear),
            ],
            Hawk
        ),
        7
    );
}

/// Ports legacy `d_pair_with_three_types_caps`: four unique types cap at 9.
#[test]
fn hawk_d_pair_with_four_types_caps_at_nine() {
    assert_eq!(
        wildlife_score(
            &[
                (0, 0, Hawk),
                (5, 0, Hawk),
                (1, 0, Bear),
                (2, 0, Elk),
                (3, 0, Salmon),
                (4, 0, Fox),
            ],
            Hawk
        ),
        9
    );
}

/// Ports legacy `d_each_hawk_in_at_most_one_pair` (and matching.rs
/// `single_edge`): two 4-point candidate pairs share the middle hawk, so only
/// one can score.
#[test]
fn hawk_d_shared_hawk_allows_single_pair() {
    assert_eq!(
        wildlife_score(
            &[
                (0, 0, Hawk),
                (2, 0, Hawk),
                (4, 0, Hawk),
                (1, 0, Bear),
                (3, 0, Elk),
            ],
            Hawk
        ),
        4
    );
}

/// Ports legacy `d_two_disjoint_pairs` (and matching.rs `two_disjoint_pairs`):
/// a 7-point pair and a 4-point pair with no shared hawks sum to 11. The
/// cross-axis hawk pairs all have empty cells between them (0 types), so no
/// spurious edges exist.
#[test]
fn hawk_d_two_disjoint_pairs_sum_to_eleven() {
    assert_eq!(
        wildlife_score(
            &[
                (0, 0, Hawk),
                (3, 0, Hawk),
                (1, 0, Bear),
                (2, 0, Elk),
                (0, 3, Hawk),
                (3, 3, Hawk),
                (1, 3, Salmon),
            ],
            Hawk
        ),
        11
    );
}

/// A hawk strictly between two others blocks their line of sight; only the
/// unblocked 2-type pair scores.
#[test]
fn hawk_d_blocker_hawk_cuts_line_of_sight() {
    assert_eq!(
        wildlife_score(
            &[
                (0, 0, Hawk),
                (2, 0, Hawk),
                (5, 0, Hawk),
                (3, 0, Bear),
                (4, 0, Elk),
            ],
            Hawk
        ),
        7
    );
}

/// Ports matching.rs `picks_best_disjoint_pair`: three hawks in mutual line
/// of sight form a triangle of candidate edges (weights 7, 4, 4) that all
/// share hawks, so the matching takes the single heaviest edge = 7.
#[test]
fn hawk_d_matching_triangle_picks_heaviest_edge() {
    assert_eq!(
        wildlife_score(
            &[
                (0, 0, Hawk),
                (4, 0, Hawk),
                (0, 4, Hawk),
                (1, 0, Bear),
                (2, 0, Elk),
                (0, 1, Salmon),
                (3, 1, Bear),
            ],
            Hawk
        ),
        7
    );
}

/// Ports matching.rs `chain_of_four_picks_best_partition`, middle-heavy
/// variant: edge weights 4 / 9 / 4 along a hawk chain; the single 9-point
/// middle pair beats the two 4-point outer pairs (9 > 8).
#[test]
fn hawk_d_matching_chain_prefers_heavy_middle_pair() {
    assert_eq!(
        wildlife_score(
            &[
                (0, 0, Hawk),
                (3, 0, Hawk),
                (7, 0, Hawk),
                (10, 0, Hawk),
                (1, 0, Bear),
                (4, 0, Bear),
                (5, 0, Elk),
                (6, 0, Salmon),
                (8, 0, Bear),
            ],
            Hawk
        ),
        9
    );
}

/// Chain variant where the outer pairs win: edge weights 4 / 7 / 4; the two
/// disjoint outer pairs (8) beat the middle pair (7).
#[test]
fn hawk_d_matching_chain_prefers_outer_pairs_when_better() {
    assert_eq!(
        wildlife_score(
            &[
                (0, 0, Hawk),
                (3, 0, Hawk),
                (6, 0, Hawk),
                (9, 0, Hawk),
                (1, 0, Bear),
                (4, 0, Bear),
                (5, 0, Elk),
                (7, 0, Salmon),
            ],
            Hawk
        ),
        8
    );
}

/// Ports matching.rs `six_node_clique`: six hawks in a line with one animal
/// in every gap; the matching takes the three disjoint 4-point pairs = 12.
#[test]
fn hawk_d_matching_three_disjoint_pairs_score_twelve() {
    assert_eq!(
        wildlife_score(
            &[
                (0, 0, Hawk),
                (2, 0, Hawk),
                (4, 0, Hawk),
                (6, 0, Hawk),
                (8, 0, Hawk),
                (10, 0, Hawk),
                (1, 0, Bear),
                (3, 0, Elk),
                (5, 0, Salmon),
                (7, 0, Bear),
                (9, 0, Elk),
            ],
            Hawk
        ),
        12
    );
}

// =====================================================================
// Fox B: each fox counts the unique non-fox wildlife types appearing at
// least twice among its 6 neighbors: 0 = 0, 1 = 3, 2 = 5, 3+ = 7.
// =====================================================================

#[test]
fn fox_b_lone_fox_scores_zero() {
    assert_eq!(wildlife_score(&[(0, 0, Fox)], Fox), 0);
}

/// Ports legacy `b_no_pairs`: three unique single neighbors, no type twice.
#[test]
fn fox_b_unique_singles_score_zero() {
    assert_eq!(
        wildlife_score(
            &[(0, 0, Fox), (1, 0, Bear), (-1, 0, Elk), (0, 1, Salmon)],
            Fox
        ),
        0
    );
}

/// Ports legacy `b_one_pair_type`: two bears = 3.
#[test]
fn fox_b_one_pair_type_scores_three() {
    assert_eq!(
        wildlife_score(&[(0, 0, Fox), (1, 0, Bear), (-1, 0, Bear)], Fox),
        3
    );
}

#[test]
fn fox_b_two_pair_types_score_five() {
    assert_eq!(
        wildlife_score(
            &[
                (0, 0, Fox),
                (1, 0, Bear),
                (-1, 0, Bear),
                (0, 1, Elk),
                (0, -1, Elk),
            ],
            Fox
        ),
        5
    );
}

/// Ports legacy `b_three_pair_types`: 2 bear + 2 elk + 2 salmon around one
/// fox = 7 (the physical cap with 6 neighbors).
#[test]
fn fox_b_three_pair_types_score_seven() {
    assert_eq!(
        wildlife_score(
            &[
                (0, 0, Fox),
                (1, 0, Bear),
                (-1, 0, Bear),
                (0, 1, Elk),
                (0, -1, Elk),
                (1, -1, Salmon),
                (-1, 1, Salmon),
            ],
            Fox
        ),
        7
    );
}

/// Ports legacy `b_excludes_fox_neighbors`: adjacent foxes never form a
/// pair type.
#[test]
fn fox_b_adjacent_foxes_do_not_count() {
    assert_eq!(
        wildlife_score(&[(0, 0, Fox), (1, 0, Fox), (-1, 0, Fox)], Fox),
        0
    );
}

/// Types appearing once alongside real pairs add nothing: 3 bears + 2 elk +
/// 1 salmon = 2 pair types = 5.
#[test]
fn fox_b_extra_odd_tokens_do_not_add() {
    assert_eq!(
        wildlife_score(
            &[
                (0, 0, Fox),
                (1, 0, Bear),
                (1, -1, Bear),
                (0, -1, Bear),
                (-1, 0, Elk),
                (-1, 1, Elk),
                (0, 1, Salmon),
            ],
            Fox
        ),
        5
    );
}

/// Each fox scores independently: one fox with a bear pair, another with an
/// elk pair = 3 + 3.
#[test]
fn fox_b_two_foxes_sum_independently() {
    assert_eq!(
        wildlife_score(
            &[
                (0, 0, Fox),
                (1, 0, Bear),
                (1, -1, Bear),
                (0, 3, Fox),
                (0, 2, Elk),
                (1, 2, Elk),
            ],
            Fox
        ),
        6
    );
}
