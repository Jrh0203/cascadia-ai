use std::collections::{BTreeMap, BTreeSet};

use cascadia_data::{
    BOARD_ENTITY_SIZE, BOARD_SLOTS, MARKET_ENTITY_SIZE, MAX_BOARD_TILES, PositionRecord, TARGET_DIM,
};
use cascadia_game::{Board, HexCoord, Rotation, Terrain, Tile, TileId, Wildlife, WildlifeMask};
use serde::{Deserialize, Serialize};

use crate::{ABLATIONS, AdaptiveMultiResolutionState, FeatureAblation, NearFieldRadius, Result};

pub const ADVERSARIAL_SCHEMA: &str = "r4-adaptive-multires-adversarial-v1";
pub const ADVERSARIAL_PARITY_SCHEMA: &str = "r4-adaptive-multires-adversarial-parity-v1";

const NONE: u8 = u8::MAX;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdversarialFixtureId {
    FarHabitatComponent,
    LongSalmonTopology,
    FarHawkConflict,
    FarFoxDiversity,
    FarLegalFrontier,
    OverflowConsequence,
    RelativeOpponentBoard,
}

impl AdversarialFixtureId {
    pub const ALL: [Self; 7] = [
        Self::FarHabitatComponent,
        Self::LongSalmonTopology,
        Self::FarHawkConflict,
        Self::FarFoxDiversity,
        Self::FarLegalFrontier,
        Self::OverflowConsequence,
        Self::RelativeOpponentBoard,
    ];

    pub const fn id(self) -> &'static str {
        match self {
            Self::FarHabitatComponent => "far-habitat-component",
            Self::LongSalmonTopology => "long-salmon-topology",
            Self::FarHawkConflict => "far-hawk-conflict",
            Self::FarFoxDiversity => "far-fox-diversity",
            Self::FarLegalFrontier => "far-legal-frontier",
            Self::OverflowConsequence => "overflow-consequence",
            Self::RelativeOpponentBoard => "relative-opponent-board",
        }
    }

    pub const fn registered_resolver(self) -> FeatureAblation {
        match self {
            Self::FarHabitatComponent => FeatureAblation::Habitat,
            Self::LongSalmonTopology
            | Self::FarHawkConflict
            | Self::FarFoxDiversity
            | Self::RelativeOpponentBoard => FeatureAblation::Wildlife,
            Self::FarLegalFrontier | Self::OverflowConsequence => FeatureAblation::Frontier,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdversarialAblationComparison {
    pub ablation_id: String,
    pub left_blake3: String,
    pub right_blake3: String,
    pub distinguishes: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdversarialCaseResult {
    pub fixture_id: String,
    pub radius_id: String,
    pub registered_resolver: String,
    pub scalar_controls_match: bool,
    pub exact_authority_distinct: bool,
    pub codec_round_trip_passed: bool,
    pub near_only_collision_passed: bool,
    pub registered_resolver_passed: bool,
    pub all_topology_passed: bool,
    pub exact_far_control_passed: bool,
    pub left_exact_blake3: String,
    pub right_exact_blake3: String,
    pub ablations: Vec<AdversarialAblationComparison>,
    pub passed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdversarialScientific {
    pub schema: String,
    pub experiment_id: String,
    pub fixture_count: usize,
    pub case_count: usize,
    pub cases: Vec<AdversarialCaseResult>,
    pub passed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdversarialReport {
    pub scientific: AdversarialScientific,
    pub scientific_blake3: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdversarialParityScientific {
    pub schema: String,
    pub experiment_id: String,
    pub report_count: usize,
    pub report_scientific_blake3s: Vec<String>,
    pub all_scientific_reports_identical: bool,
    pub all_suites_passed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdversarialParityReport {
    pub scientific: AdversarialParityScientific,
    pub scientific_blake3: String,
}

pub fn run_adversarial_suite() -> Result<AdversarialReport> {
    let mut cases =
        Vec::with_capacity(AdversarialFixtureId::ALL.len() * NearFieldRadius::ALL.len());
    for fixture in AdversarialFixtureId::ALL {
        cases.extend(evaluate_adversarial_fixture(fixture)?);
    }
    let passed = cases.iter().all(|case| case.passed);
    let scientific = AdversarialScientific {
        schema: ADVERSARIAL_SCHEMA.to_owned(),
        experiment_id: crate::census::EXPERIMENT_ID.to_owned(),
        fixture_count: AdversarialFixtureId::ALL.len(),
        case_count: cases.len(),
        cases,
        passed,
    };
    let scientific_blake3 = blake3::hash(&serde_json::to_vec(&scientific)?)
        .to_hex()
        .to_string();
    Ok(AdversarialReport {
        scientific,
        scientific_blake3,
    })
}

pub fn validate_adversarial_report(report: &AdversarialReport) -> Result<()> {
    let scientific_blake3 = blake3::hash(&serde_json::to_vec(&report.scientific)?)
        .to_hex()
        .to_string();
    if report.scientific.schema != ADVERSARIAL_SCHEMA
        || report.scientific.experiment_id != crate::census::EXPERIMENT_ID
        || report.scientific.fixture_count != AdversarialFixtureId::ALL.len()
        || report.scientific.case_count
            != AdversarialFixtureId::ALL.len() * NearFieldRadius::ALL.len()
        || scientific_blake3 != report.scientific_blake3
    {
        return Err(crate::R4Error::AggregateContract(
            "adversarial report contract or scientific hash drifted".to_owned(),
        ));
    }
    Ok(())
}

pub fn compare_adversarial_reports(
    reports: &[AdversarialReport],
) -> Result<AdversarialParityReport> {
    if reports.is_empty() {
        return Err(crate::R4Error::AggregateContract(
            "adversarial parity requires at least one report".to_owned(),
        ));
    }
    for report in reports {
        validate_adversarial_report(report)?;
    }
    let report_scientific_blake3s = reports
        .iter()
        .map(|report| report.scientific_blake3.clone())
        .collect::<Vec<_>>();
    let all_scientific_reports_identical = report_scientific_blake3s
        .windows(2)
        .all(|pair| pair[0] == pair[1]);
    let all_suites_passed = reports.iter().all(|report| report.scientific.passed);
    let scientific = AdversarialParityScientific {
        schema: ADVERSARIAL_PARITY_SCHEMA.to_owned(),
        experiment_id: crate::census::EXPERIMENT_ID.to_owned(),
        report_count: reports.len(),
        report_scientific_blake3s,
        all_scientific_reports_identical,
        all_suites_passed,
    };
    let scientific_blake3 = blake3::hash(&serde_json::to_vec(&scientific)?)
        .to_hex()
        .to_string();
    Ok(AdversarialParityReport {
        scientific,
        scientific_blake3,
    })
}

pub fn evaluate_adversarial_fixture(
    fixture: AdversarialFixtureId,
) -> Result<Vec<AdversarialCaseResult>> {
    let (left, right) = adversarial_fixture_pair(fixture);
    let scalar_controls_match =
        left.wildlife_counts == right.wildlife_counts && left.habitat_sizes == right.habitat_sizes;
    let resolver = fixture.registered_resolver();
    NearFieldRadius::ALL
        .into_iter()
        .map(|radius| {
            let left_state =
                AdaptiveMultiResolutionState::from_position_record(&left, None, radius)?;
            let right_state =
                AdaptiveMultiResolutionState::from_position_record(&right, None, radius)?;
            let left_exact = left_state.to_packed_bytes()?;
            let right_exact = right_state.to_packed_bytes()?;
            let codec_round_trip_passed =
                AdaptiveMultiResolutionState::from_packed_bytes(&left_exact)? == left_state
                    && AdaptiveMultiResolutionState::from_packed_bytes(&right_exact)?
                        == right_state;
            let mut ablations = Vec::with_capacity(ABLATIONS.len());
            for arm in ABLATIONS {
                let left_bytes = left_state.feature_view(arm)?.canonical_bytes()?;
                let right_bytes = right_state.feature_view(arm)?.canonical_bytes()?;
                ablations.push(AdversarialAblationComparison {
                    ablation_id: arm.id().to_owned(),
                    left_blake3: blake3::hash(&left_bytes).to_hex().to_string(),
                    right_blake3: blake3::hash(&right_bytes).to_hex().to_string(),
                    distinguishes: left_bytes != right_bytes,
                });
            }
            let distinguishes = |arm: FeatureAblation| {
                ablations
                    .iter()
                    .find(|comparison| comparison.ablation_id == arm.id())
                    .expect("every frozen ablation was evaluated")
                    .distinguishes
            };
            let near_only_collision_passed = !distinguishes(FeatureAblation::NearOnly);
            let registered_resolver_passed = distinguishes(resolver);
            let all_topology_passed = distinguishes(FeatureAblation::AllTopology);
            let exact_far_control_passed = distinguishes(FeatureAblation::ExactFarControl);
            let exact_authority_distinct = left_exact != right_exact;
            let passed = scalar_controls_match
                && exact_authority_distinct
                && codec_round_trip_passed
                && near_only_collision_passed
                && registered_resolver_passed
                && all_topology_passed
                && exact_far_control_passed;
            Ok(AdversarialCaseResult {
                fixture_id: fixture.id().to_owned(),
                radius_id: radius.id().to_owned(),
                registered_resolver: resolver.id().to_owned(),
                scalar_controls_match,
                exact_authority_distinct,
                codec_round_trip_passed,
                near_only_collision_passed,
                registered_resolver_passed,
                all_topology_passed,
                exact_far_control_passed,
                left_exact_blake3: blake3::hash(&left_exact).to_hex().to_string(),
                right_exact_blake3: blake3::hash(&right_exact).to_hex().to_string(),
                ablations,
                passed,
            })
        })
        .collect()
}

pub fn adversarial_fixture_pair(fixture: AdversarialFixtureId) -> (PositionRecord, PositionRecord) {
    match fixture {
        AdversarialFixtureId::FarHabitatComponent => habitat_pair(),
        AdversarialFixtureId::LongSalmonTopology => salmon_pair(),
        AdversarialFixtureId::FarHawkConflict => hawk_pair(),
        AdversarialFixtureId::FarFoxDiversity => fox_pair(),
        AdversarialFixtureId::FarLegalFrontier => frontier_pair(),
        AdversarialFixtureId::OverflowConsequence => overflow_pair(),
        AdversarialFixtureId::RelativeOpponentBoard => opponent_pair(),
    }
}

#[derive(Debug, Clone)]
struct BoardSpec {
    coordinates: Vec<HexCoord>,
    tiles: Vec<Tile>,
    rotations: BTreeMap<HexCoord, Rotation>,
    wildlife: BTreeMap<HexCoord, Wildlife>,
}

impl BoardSpec {
    fn straight_single(terrain: Terrain, wildlife: Wildlife) -> Self {
        let coordinates = straight_coordinates();
        let tiles = coordinates
            .iter()
            .enumerate()
            .map(|(index, _)| single_tile(index, terrain, wildlife))
            .collect();
        Self {
            coordinates,
            tiles,
            rotations: BTreeMap::new(),
            wildlife: BTreeMap::new(),
        }
    }
}

fn straight_coordinates() -> Vec<HexCoord> {
    (-11..=11).map(|q| HexCoord::new(q, 0)).collect()
}

fn overflow_coordinates(extra: HexCoord) -> Vec<HexCoord> {
    let mut coordinates = vec![HexCoord::ORIGIN];
    for (dq, dr) in [(1, 0), (0, -1), (-1, 1)] {
        coordinates.extend((1..=7).map(|distance| HexCoord::new(dq * distance, dr * distance)));
    }
    coordinates.push(extra);
    assert_eq!(coordinates.len(), 23);
    assert_eq!(
        coordinates.iter().copied().collect::<BTreeSet<_>>().len(),
        23
    );
    coordinates
}

fn single_tile(index: usize, terrain: Terrain, wildlife: Wildlife) -> Tile {
    Tile {
        id: TileId(index as u8),
        terrain_a: terrain,
        terrain_b: None,
        wildlife: WildlifeMask::one(wildlife),
        keystone: true,
    }
}

fn dual_tile(index: usize) -> Tile {
    Tile {
        id: TileId(index as u8),
        terrain_a: Terrain::Forest,
        terrain_b: Some(Terrain::River),
        wildlife: WildlifeMask::from_bits(0b1_1111),
        keystone: false,
    }
}

fn terminal_record(boards: Vec<BoardSpec>) -> PositionRecord {
    let player_count = boards.len() as u8;
    let total_turns = 20 * player_count;
    let mut record = PositionRecord {
        game_index: 0x7777,
        turn: total_turns,
        active_seat: 0,
        player_count,
        total_turns,
        board_counts: [0; BOARD_SLOTS],
        nature_tokens: [0; BOARD_SLOTS],
        scoring_cards: [0; 5],
        habitat_bonuses: false,
        wildlife_counts: [[0; 5]; BOARD_SLOTS],
        habitat_sizes: [[0; 5]; BOARD_SLOTS],
        board_entities: [[[NONE; BOARD_ENTITY_SIZE]; MAX_BOARD_TILES]; BOARD_SLOTS],
        market_entities: [[NONE; MARKET_ENTITY_SIZE]; 4],
        targets: [0; TARGET_DIM],
    };
    for (seat, spec) in boards.into_iter().enumerate() {
        assert_eq!(spec.coordinates.len(), 23);
        assert_eq!(spec.tiles.len(), 23);
        assert_eq!(
            spec.coordinates
                .iter()
                .copied()
                .collect::<BTreeSet<_>>()
                .len(),
            23
        );
        let mut board = Board::empty();
        for (coord, tile) in spec
            .coordinates
            .iter()
            .copied()
            .zip(spec.tiles.iter().copied())
        {
            board
                .place_tile(
                    coord,
                    tile,
                    spec.rotations
                        .get(&coord)
                        .copied()
                        .unwrap_or(Rotation::ZERO),
                )
                .expect("adversarial fixtures are legal connected boards");
        }
        for (coord, wildlife) in &spec.wildlife {
            board
                .place_wildlife(*coord, *wildlife)
                .expect("adversarial wildlife placements are legal");
        }
        board.validate().expect("adversarial board is valid");
        record.board_counts[seat] = 23;
        for terrain in Terrain::ALL {
            record.habitat_sizes[seat][terrain as usize] = board.largest_habitat(terrain);
        }
        for wildlife in spec.wildlife.values() {
            record.wildlife_counts[seat][*wildlife as usize] += 1;
        }
        let mut placed = board.placed_tiles().collect::<Vec<_>>();
        placed.sort_unstable_by_key(|(coord, _)| (coord.q, coord.r));
        for (row, (coord, tile)) in placed.into_iter().enumerate() {
            record.board_entities[seat][row] = [
                coord.q as u8,
                coord.r as u8,
                tile.tile.terrain_a as u8,
                tile.tile.terrain_b.map_or(NONE, |terrain| terrain as u8),
                tile.rotation.get(),
                tile.tile.wildlife.bits(),
                tile.wildlife.map_or(NONE, |wildlife| wildlife as u8),
                u8::from(tile.tile.keystone),
            ];
        }
    }
    record
}

fn habitat_pair() -> (PositionRecord, PositionRecord) {
    let left_forest = BTreeSet::from([-11, -10, -9, -8, -6, 7]);
    let right_forest = BTreeSet::from([-10, -9, -8, -7, 6, 7]);
    let make = |forest: &BTreeSet<i8>| {
        let coordinates = straight_coordinates();
        let tiles = coordinates
            .iter()
            .enumerate()
            .map(|(index, coord)| {
                if forest.contains(&coord.q) {
                    single_tile(index, Terrain::Forest, Wildlife::Bear)
                } else {
                    single_tile(index, Terrain::River, Wildlife::Salmon)
                }
            })
            .collect();
        terminal_record(vec![BoardSpec {
            coordinates,
            tiles,
            rotations: BTreeMap::new(),
            wildlife: BTreeMap::new(),
        }])
    };
    (make(&left_forest), make(&right_forest))
}

fn salmon_pair() -> (PositionRecord, PositionRecord) {
    let make = |salmon: &[i8]| {
        let mut board = BoardSpec::straight_single(Terrain::River, Wildlife::Salmon);
        board.wildlife = salmon
            .iter()
            .map(|q| (HexCoord::new(*q, 0), Wildlife::Salmon))
            .collect();
        terminal_record(vec![board])
    };
    (make(&[6, 7, 8, 9, 10]), make(&[-10, 6, 7, 8, 9]))
}

fn hawk_pair() -> (PositionRecord, PositionRecord) {
    let make = |hawks: &[i8]| {
        let mut board = BoardSpec::straight_single(Terrain::Prairie, Wildlife::Hawk);
        board.wildlife = hawks
            .iter()
            .map(|q| (HexCoord::new(*q, 0), Wildlife::Hawk))
            .collect();
        terminal_record(vec![board])
    };
    (make(&[6, 7, 10]), make(&[6, 8, 10]))
}

fn fox_pair() -> (PositionRecord, PositionRecord) {
    let make = |placements: &[(i8, Wildlife)]| {
        let coordinates = straight_coordinates();
        let tiles = coordinates
            .iter()
            .enumerate()
            .map(|(index, _)| dual_tile(index))
            .collect();
        let wildlife = placements
            .iter()
            .map(|(q, wildlife)| (HexCoord::new(*q, 0), *wildlife))
            .collect();
        terminal_record(vec![BoardSpec {
            coordinates,
            tiles,
            rotations: BTreeMap::new(),
            wildlife,
        }])
    };
    (
        make(&[
            (-9, Wildlife::Bear),
            (7, Wildlife::Bear),
            (8, Wildlife::Fox),
            (9, Wildlife::Elk),
        ]),
        make(&[
            (-9, Wildlife::Elk),
            (7, Wildlife::Bear),
            (8, Wildlife::Fox),
            (9, Wildlife::Bear),
        ]),
    )
}

fn frontier_pair() -> (PositionRecord, PositionRecord) {
    let make = |rotation: Rotation| {
        let coordinates = straight_coordinates();
        let tiles = coordinates
            .iter()
            .enumerate()
            .map(|(index, coord)| {
                if coord.q == 10 {
                    dual_tile(index)
                } else {
                    single_tile(index, Terrain::River, Wildlife::Salmon)
                }
            })
            .collect();
        terminal_record(vec![BoardSpec {
            coordinates,
            tiles,
            rotations: BTreeMap::from([(HexCoord::new(10, 0), rotation)]),
            wildlife: BTreeMap::new(),
        }])
    };
    (make(Rotation::ZERO), make(Rotation::FIVE))
}

fn overflow_pair() -> (PositionRecord, PositionRecord) {
    let make = |extra| {
        let coordinates = overflow_coordinates(extra);
        let tiles = coordinates
            .iter()
            .enumerate()
            .map(|(index, _)| single_tile(index, Terrain::River, Wildlife::Salmon))
            .collect();
        terminal_record(vec![BoardSpec {
            coordinates,
            tiles,
            rotations: BTreeMap::new(),
            wildlife: BTreeMap::new(),
        }])
    };
    (make(HexCoord::new(7, -1)), make(HexCoord::new(1, -7)))
}

fn opponent_pair() -> (PositionRecord, PositionRecord) {
    let focal = BoardSpec::straight_single(Terrain::River, Wildlife::Salmon);
    let make_opponent = |salmon: &[i8]| {
        let mut board = BoardSpec::straight_single(Terrain::River, Wildlife::Salmon);
        board.wildlife = salmon
            .iter()
            .map(|q| (HexCoord::new(*q, 0), Wildlife::Salmon))
            .collect();
        board
    };
    (
        terminal_record(vec![focal.clone(), make_opponent(&[6, 7, 8, 9, 10])]),
        terminal_record(vec![focal, make_opponent(&[-10, 6, 7, 8, 9])]),
    )
}
