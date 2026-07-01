use std::{collections::BTreeMap, sync::OnceLock};

use blake3::Hasher;
use cascadia_ai::{
    draft_opponents::{
        preference_draft_move, random_draft_move, sample_preferences, scarcity_draft_move,
    },
    nnue::{
        BagInfo, LEGACY_MID_V4_FIXED_V1_OPP_BASE, LEGACY_MID_V4_FIXED_V1_OPP_END,
        LEGACY_MID_V4_FIXED_V1_OVERFLOW_BASE, LEGACY_MID_V4_FIXED_V1_SCHEMA_ID,
        LEGACY_MID_V4_FIXED_V1_TAIL_FEATURES, LEGACY_MID_V4_FIXED_V1_TBAG_TERRAIN_BASE,
        LEGACY_MID_V4_FIXED_V1_TBAG_WL_BASE, NUM_FEATURES, NUM_FEATURES_LEGACY_MID_V4_FIXED_V1,
        TBAG_EXT_BINS, extract_features_with_bag,
    },
    search::{execute_scored_move, greedy_move},
};
use cascadia_core::{
    board::Board,
    game::GameState,
    hex::HexCoord,
    market::{Market, MarketPair, TileBag},
    types::{ScoringCards, Terrain, TileData, Wildlife, WildlifeMask},
};
use rand::{SeedableRng, rngs::StdRng};
use serde::{Deserialize, Serialize};

use crate::{Result, canonical_blake3, invalid, update_framed};

pub const EXPERIMENT_ID: &str = "corrected-mid-tail-activation-census-v1";
pub const FEATURE_SCHEMA: &str = "legacy-mid-v4-fixed-v1";
pub const DATASET_ID: &str = "corrected-mid-tail-public-state-corpus-v1";
pub const ARTIFACT_SCHEMA_VERSION: u32 = 2;
pub const PLAYERS: usize = 4;
pub const TURNS_PER_PLAYER: usize = 20;
pub const ROWS_PER_GAME: usize = PLAYERS * TURNS_PER_PLAYER;
pub const PRODUCTION_FIRST_GAME_INDEX: u64 = 0;
pub const PRODUCTION_TOTAL_GAMES: usize = 1_024;
pub const PRODUCTION_SHARD_COUNT: usize = 4;
pub const OVERFLOW_WITNESS_MAX_SEARCH: u64 = 100_000;

const GAME_SEED_DOMAIN: &[u8] = b"corrected-mid-tail-activation-census-v1/game-seed/v1";
const POLICY_SEED_DOMAIN: &[u8] = b"corrected-mid-tail-activation-census-v1/policy-seed/v1";
const OVERFLOW_WITNESS_DOMAIN: &[u8] =
    b"corrected-mid-tail-activation-census-v1/reachable-overflow-witness/v1";
const FEATURE_RECEIPT_DOMAIN: &[u8] = b"corrected-mid-tail-activation-census-v1/features/v1";
const TAIL_RECEIPT_DOMAIN: &[u8] = b"corrected-mid-tail-activation-census-v1/corrected-tail/v1";

const FORBIDDEN_ENVIRONMENT: &[&str] = &[
    "CASCADIA_GREEDY_POTENTIAL",
    "CASCADIA_MCE_CACHE",
    "CASCADIA_POTENTIAL_DIAGNOSTICS",
    "LEGACY_TEACHER_GREEDY_ASSERT_PARITY",
    "LEGACY_TEACHER_GREEDY_REFERENCE",
    "LEGACY_TEACHER_POTENTIAL_FULL_RECOMPUTE",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    Opening,
    Early,
    Middle,
    Late,
}

impl Phase {
    pub const ALL: [Self; 4] = [Self::Opening, Self::Early, Self::Middle, Self::Late];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Opening => "opening",
            Self::Early => "early",
            Self::Middle => "middle",
            Self::Late => "late",
        }
    }

    pub fn index(self) -> usize {
        match self {
            Self::Opening => 0,
            Self::Early => 1,
            Self::Middle => 2,
            Self::Late => 3,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrajectoryPolicy {
    Greedy,
    RandomDraft,
    ScarcityDraft,
    PreferenceDraft,
}

impl TrajectoryPolicy {
    pub const ALL: [Self; 4] = [
        Self::Greedy,
        Self::RandomDraft,
        Self::ScarcityDraft,
        Self::PreferenceDraft,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Greedy => "greedy",
            Self::RandomDraft => "random_draft",
            Self::ScarcityDraft => "scarcity_draft",
            Self::PreferenceDraft => "preference_draft",
        }
    }

    pub fn id(self) -> usize {
        match self {
            Self::Greedy => 0,
            Self::RandomDraft => 1,
            Self::ScarcityDraft => 2,
            Self::PreferenceDraft => 3,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecordKind {
    Representative,
    ReachableOverflowWitness,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecordProvenance {
    pub kind: RecordKind,
    pub game_index: Option<u64>,
    pub decision_index: u8,
    pub policy: TrajectoryPolicy,
    pub fixture_search_counter: Option<u64>,
    pub seed_commitment_blake3: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PublicTile {
    pub placement_ordinal: u8,
    pub q: i8,
    pub r: i8,
    pub rotation: u8,
    pub terrain1: u8,
    pub terrain2: Option<u8>,
    pub allowed_mask: u8,
    pub keystone: bool,
    pub wildlife: Option<u8>,
    pub starter: bool,
}

impl PublicTile {
    fn from_board(board: &Board, placement_ordinal: usize, index: usize) -> Result<Self> {
        let cell = board.grid.get(index);
        if !cell.is_present() {
            return Err(invalid("placed tile index points to an empty board cell"));
        }
        let coordinate = HexCoord::from_index(index);
        Ok(Self {
            placement_ordinal: u8::try_from(placement_ordinal)?,
            q: coordinate.q,
            r: coordinate.r,
            rotation: board.rotations[index],
            terrain1: cell
                .primary_terrain()
                .ok_or_else(|| invalid("present tile has no primary terrain"))?
                as u8,
            terrain2: cell.secondary_terrain().map(|terrain| terrain as u8),
            allowed_mask: cell.allowed_wildlife().0,
            keystone: cell.is_keystone(),
            wildlife: cell.placed_wildlife().map(|wildlife| wildlife as u8),
            starter: placement_ordinal < 3,
        })
    }

    fn tile_data(&self) -> Result<TileData> {
        let terrain1 = Terrain::from_u8(self.terrain1)
            .ok_or_else(|| invalid("public tile primary terrain is out of range"))?;
        let terrain2 = self
            .terrain2
            .map(|value| {
                Terrain::from_u8(value)
                    .ok_or_else(|| invalid("public tile secondary terrain is out of range"))
            })
            .transpose()?;
        if self.rotation >= 6 {
            return Err(invalid("public tile rotation is out of range"));
        }
        if self.allowed_mask == 0 || self.allowed_mask & !0x1f != 0 {
            return Err(invalid("public tile wildlife mask is invalid"));
        }
        if self.keystone != terrain2.is_none() {
            return Err(invalid(
                "public tile keystone flag disagrees with single/dual terrain shape",
            ));
        }
        Ok(TileData {
            terrain1,
            terrain2,
            allowed: WildlifeMask(self.allowed_mask),
            keystone: self.keystone,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PublicBoard {
    pub absolute_seat: u8,
    pub nature_tokens: u8,
    pub tiles: Vec<PublicTile>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PublicMarketPair {
    pub terrain1: u8,
    pub terrain2: Option<u8>,
    pub allowed_mask: u8,
    pub keystone: bool,
    pub wildlife: u8,
}

impl PublicMarketPair {
    fn from_market(pair: MarketPair) -> Self {
        Self {
            terrain1: pair.tile.terrain1 as u8,
            terrain2: pair.tile.terrain2.map(|terrain| terrain as u8),
            allowed_mask: pair.tile.allowed.0,
            keystone: pair.tile.keystone,
            wildlife: pair.wildlife as u8,
        }
    }

    fn market_pair(self) -> Result<MarketPair> {
        let tile = PublicTile {
            placement_ordinal: 0,
            q: 0,
            r: 0,
            rotation: 0,
            terrain1: self.terrain1,
            terrain2: self.terrain2,
            allowed_mask: self.allowed_mask,
            keystone: self.keystone,
            wildlife: None,
            starter: false,
        }
        .tile_data()?;
        let wildlife = Wildlife::from_u8(self.wildlife)
            .ok_or_else(|| invalid("public market wildlife is out of range"))?;
        Ok(MarketPair { tile, wildlife })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PublicSupply {
    pub remaining_tiles: u8,
    pub terrain_counts: [u8; 5],
    pub wildlife_capacity_counts: [u8; 5],
    pub wildlife_remaining: [u8; 5],
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PublicState {
    pub ruleset: String,
    pub num_players: u8,
    pub focal_seat: u8,
    pub personal_turn: u8,
    pub phase: Phase,
    pub turns_remaining: u8,
    pub free_overflow_applied: bool,
    pub overflow_used_this_turn: bool,
    pub boards: Vec<PublicBoard>,
    pub market: [Option<PublicMarketPair>; 4],
    pub supply: PublicSupply,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PublicStateRecord {
    pub schema_version: u32,
    pub experiment_id: String,
    pub feature_schema: String,
    pub provenance: RecordProvenance,
    pub state: PublicState,
    pub public_state_blake3: String,
    pub raw_feature_emissions: u16,
    pub normalized_feature_activations: u16,
    pub normalized_features_blake3: String,
    pub corrected_tail_features: Vec<u16>,
    pub corrected_tail_blake3: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayedRecord {
    pub raw_feature_emissions: usize,
    pub normalized_features: Vec<u16>,
    pub corrected_tail_features: Vec<u16>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct TileKey {
    terrain1: u8,
    terrain2: Option<u8>,
    allowed_mask: u8,
    keystone: bool,
}

impl TileKey {
    fn from_tile(tile: TileData) -> Self {
        Self {
            terrain1: tile.terrain1 as u8,
            terrain2: tile.terrain2.map(|terrain| terrain as u8),
            allowed_mask: tile.allowed.0,
            keystone: tile.keystone,
        }
    }

    fn from_public(tile: &PublicTile) -> Result<Self> {
        Ok(Self::from_tile(tile.tile_data()?))
    }

    fn from_market(pair: PublicMarketPair) -> Result<Self> {
        Ok(Self::from_tile(pair.market_pair()?.tile))
    }
}

#[derive(Debug)]
struct PublicSupplyProjection {
    supply: PublicSupply,
    tbag_joint: [[u8; 5]; 5],
}

pub fn validate_compiled_contract() -> Result<()> {
    if LEGACY_MID_V4_FIXED_V1_SCHEMA_ID != FEATURE_SCHEMA
        || NUM_FEATURES != 11_231
        || NUM_FEATURES != NUM_FEATURES_LEGACY_MID_V4_FIXED_V1
        || LEGACY_MID_V4_FIXED_V1_OPP_BASE != 10_561
        || LEGACY_MID_V4_FIXED_V1_OPP_END != 10_930
        || LEGACY_MID_V4_FIXED_V1_TBAG_TERRAIN_BASE != 10_930
        || LEGACY_MID_V4_FIXED_V1_TBAG_WL_BASE != 11_080
        || LEGACY_MID_V4_FIXED_V1_OVERFLOW_BASE != 11_230
        || LEGACY_MID_V4_FIXED_V1_TAIL_FEATURES != 301
        || TBAG_EXT_BINS != 30
    {
        return Err(invalid(
            "compiled corrected extractor constants do not match ADR 0137",
        ));
    }
    Ok(())
}

pub fn validate_generation_environment() -> Result<()> {
    let configured = FORBIDDEN_ENVIRONMENT
        .iter()
        .filter(|name| std::env::var_os(name).is_some())
        .copied()
        .collect::<Vec<_>>();
    if !configured.is_empty() {
        return Err(invalid(format!(
            "trajectory-affecting environment variables must be unset: {}",
            configured.join(", ")
        )));
    }
    Ok(())
}

pub fn phase_for_turn(personal_turn: u8) -> Result<Phase> {
    match personal_turn {
        1 => Ok(Phase::Opening),
        2..=5 => Ok(Phase::Early),
        6..=13 => Ok(Phase::Middle),
        14..=20 => Ok(Phase::Late),
        _ => Err(invalid("personal turn must be in 1..=20")),
    }
}

pub fn policy_for(game_index: u64, seat: usize) -> TrajectoryPolicy {
    TrajectoryPolicy::ALL[((game_index as usize) + seat) % TrajectoryPolicy::ALL.len()]
}

pub fn owned_game_indices(
    first_game_index: u64,
    total_games: usize,
    shard_index: usize,
    shard_count: usize,
) -> Result<Vec<u64>> {
    if total_games == 0 || shard_count == 0 || shard_index >= shard_count {
        return Err(invalid(
            "total games and shard count must be positive and shard index must be in range",
        ));
    }
    let end = first_game_index
        .checked_add(u64::try_from(total_games)?)
        .ok_or_else(|| invalid("game range overflows u64"))?;
    Ok((first_game_index..end)
        .filter(|index| {
            ((*index - first_game_index) % u64::try_from(shard_count).unwrap())
                == u64::try_from(shard_index).unwrap()
        })
        .collect())
}

pub(crate) fn generate_representative_game(game_index: u64) -> Result<Vec<PublicStateRecord>> {
    validate_compiled_contract()?;
    validate_generation_environment()?;
    let seed = derive_seed(GAME_SEED_DOMAIN, &[game_index]);
    let mut setup_rng = StdRng::seed_from_u64(seed);
    let mut game = GameState::new(PLAYERS, ScoringCards::all_a(), &mut setup_rng);
    let policies: [TrajectoryPolicy; PLAYERS] =
        std::array::from_fn(|seat| policy_for(game_index, seat));
    let mut policy_rngs: [StdRng; PLAYERS] = std::array::from_fn(|seat| {
        StdRng::seed_from_u64(derive_seed(
            POLICY_SEED_DOMAIN,
            &[seed, seat as u64, policies[seat].id() as u64],
        ))
    });
    let preferences: [[f32; 5]; PLAYERS] = std::array::from_fn(|seat| {
        if policies[seat] == TrajectoryPolicy::PreferenceDraft {
            sample_preferences(&mut policy_rngs[seat])
        } else {
            [0.2; 5]
        }
    });
    let seed_commitment = seed_commitment(seed);
    let mut records = Vec::with_capacity(ROWS_PER_GAME);

    for decision_index in 0..ROWS_PER_GAME {
        if game.is_game_over() {
            return Err(invalid(format!(
                "representative game {game_index} ended after {decision_index} decisions"
            )));
        }
        let expected_seat = decision_index % PLAYERS;
        if game.current_player != expected_seat {
            return Err(invalid("representative trajectory seat order drifted"));
        }
        let free_overflow_applied = if game.can_replace_overflow().is_some() {
            if !game.replace_overflow() {
                return Err(invalid("legal free overflow replacement failed"));
            }
            true
        } else {
            false
        };
        let policy = policies[game.current_player];
        records.push(capture_record(
            &game,
            RecordProvenance {
                kind: RecordKind::Representative,
                game_index: Some(game_index),
                decision_index: u8::try_from(decision_index)?,
                policy,
                fixture_search_counter: None,
                seed_commitment_blake3: seed_commitment.clone(),
            },
            free_overflow_applied,
        )?);

        let seat = game.current_player;
        let movement = match policy {
            TrajectoryPolicy::Greedy => greedy_move(&game),
            TrajectoryPolicy::RandomDraft => random_draft_move(&game, &mut policy_rngs[seat]),
            TrajectoryPolicy::ScarcityDraft => scarcity_draft_move(&game, &mut policy_rngs[seat]),
            TrajectoryPolicy::PreferenceDraft => {
                preference_draft_move(&game, &preferences[seat], &mut policy_rngs[seat])
            }
        }
        .ok_or_else(|| invalid("trajectory policy produced no legal move"))?;
        if !execute_scored_move(&mut game, &movement) {
            return Err(invalid("trajectory policy produced an illegal move"));
        }
    }
    if !game.is_game_over() {
        return Err(invalid(
            "representative game remained live after exactly 80 decisions",
        ));
    }
    Ok(records)
}

pub fn generate_reachable_overflow_witness() -> Result<PublicStateRecord> {
    validate_compiled_contract()?;
    validate_generation_environment()?;
    for counter in 0..OVERFLOW_WITNESS_MAX_SEARCH {
        let seed = derive_seed(OVERFLOW_WITNESS_DOMAIN, &[counter]);
        let mut rng = StdRng::seed_from_u64(seed);
        let mut game = GameState::new(PLAYERS, ScoringCards::all_a(), &mut rng);
        if game.can_replace_overflow().is_none() {
            continue;
        }
        if !game.replace_overflow() {
            return Err(invalid("reachable overflow witness replacement failed"));
        }
        let record = capture_record(
            &game,
            RecordProvenance {
                kind: RecordKind::ReachableOverflowWitness,
                game_index: None,
                decision_index: 0,
                policy: TrajectoryPolicy::Greedy,
                fixture_search_counter: Some(counter),
                seed_commitment_blake3: seed_commitment(seed),
            },
            true,
        )?;
        if !record
            .corrected_tail_features
            .contains(&(LEGACY_MID_V4_FIXED_V1_OVERFLOW_BASE as u16))
        {
            return Err(invalid(
                "reachable overflow witness did not activate corrected overflow row",
            ));
        }
        return Ok(record);
    }
    Err(invalid(format!(
        "no reachable opening overflow found in first {OVERFLOW_WITNESS_MAX_SEARCH} fixture seeds"
    )))
}

fn capture_record(
    game: &GameState,
    provenance: RecordProvenance,
    free_overflow_applied: bool,
) -> Result<PublicStateRecord> {
    let focal_seat = game.current_player;
    let decision_index = usize::from(provenance.decision_index);
    let personal_turn = u8::try_from(decision_index / PLAYERS + 1)?;
    let boards = game
        .boards
        .iter()
        .enumerate()
        .map(|(seat, board)| {
            let tiles = board
                .placed_tiles
                .iter()
                .enumerate()
                .map(|(ordinal, index)| PublicTile::from_board(board, ordinal, usize::from(*index)))
                .collect::<Result<Vec<_>>>()?;
            Ok(PublicBoard {
                absolute_seat: u8::try_from(seat)?,
                nature_tokens: board.nature_tokens,
                tiles,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let market =
        std::array::from_fn(|index| game.market.pairs[index].map(PublicMarketPair::from_market));
    let projection = derive_public_supply(&boards, &market)?;
    let actual_bag = BagInfo::from_game_for_player(game, focal_seat);
    if projection.supply.remaining_tiles != u8::try_from(game.tile_bag.remaining())?
        || projection.supply.terrain_counts != actual_bag.tbag_terrain
        || projection.supply.wildlife_capacity_counts != actual_bag.tbag_wildlife
        || projection.supply.wildlife_remaining != actual_bag.remaining
    {
        return Err(invalid(
            "public inventory reconstruction disagrees with live game bag information",
        ));
    }
    let state = PublicState {
        ruleset: "four_player_aaaaa_no_habitat_bonus_labels".to_owned(),
        num_players: PLAYERS as u8,
        focal_seat: u8::try_from(focal_seat)?,
        personal_turn,
        phase: phase_for_turn(personal_turn)?,
        turns_remaining: game.turns_remaining,
        free_overflow_applied,
        overflow_used_this_turn: game.overflow_used_this_turn,
        boards,
        market,
        supply: projection.supply,
    };
    let public_state_blake3 = canonical_blake3(&state)?;
    let (raw_feature_emissions, normalized_features) =
        normalized_features(&game.boards[focal_seat], &actual_bag)?;
    let corrected_tail_features = corrected_tail_from(&normalized_features)?;
    let normalized_features_blake3 =
        hash_u16_features(FEATURE_RECEIPT_DOMAIN, &normalized_features);
    let corrected_tail_blake3 = hash_u16_features(TAIL_RECEIPT_DOMAIN, &corrected_tail_features);

    Ok(PublicStateRecord {
        schema_version: ARTIFACT_SCHEMA_VERSION,
        experiment_id: EXPERIMENT_ID.to_owned(),
        feature_schema: FEATURE_SCHEMA.to_owned(),
        provenance,
        state,
        public_state_blake3,
        raw_feature_emissions: u16::try_from(raw_feature_emissions)?,
        normalized_feature_activations: u16::try_from(normalized_features.len())?,
        normalized_features_blake3,
        corrected_tail_features,
        corrected_tail_blake3,
    })
}

pub fn replay_record(record: &PublicStateRecord) -> Result<ReplayedRecord> {
    validate_compiled_contract()?;
    if record.schema_version != ARTIFACT_SCHEMA_VERSION
        || record.experiment_id != EXPERIMENT_ID
        || record.feature_schema != FEATURE_SCHEMA
    {
        return Err(invalid("public state record schema identity drifted"));
    }
    validate_record_provenance(record)?;
    if record.public_state_blake3 != canonical_blake3(&record.state)? {
        return Err(invalid("public state record content hash drifted"));
    }
    validate_public_state_shape(record)?;
    let (board, bag) = reconstruct_extractor_inputs(&record.state)?;
    let (raw_feature_emissions, normalized_features) = normalized_features(&board, &bag)?;
    let corrected_tail_features = corrected_tail_from(&normalized_features)?;
    if raw_feature_emissions != usize::from(record.raw_feature_emissions)
        || normalized_features.len() != usize::from(record.normalized_feature_activations)
        || hash_u16_features(FEATURE_RECEIPT_DOMAIN, &normalized_features)
            != record.normalized_features_blake3
        || corrected_tail_features != record.corrected_tail_features
        || hash_u16_features(TAIL_RECEIPT_DOMAIN, &corrected_tail_features)
            != record.corrected_tail_blake3
    {
        return Err(invalid(
            "replayed actual Rust extractor output differs from frozen record receipts",
        ));
    }
    validate_corrected_tail_shape(
        &corrected_tail_features,
        record.state.overflow_used_this_turn,
    )?;
    Ok(ReplayedRecord {
        raw_feature_emissions,
        normalized_features,
        corrected_tail_features,
    })
}

fn validate_record_provenance(record: &PublicStateRecord) -> Result<()> {
    match record.provenance.kind {
        RecordKind::Representative => {
            let game_index = record
                .provenance
                .game_index
                .ok_or_else(|| invalid("representative record has no game index"))?;
            if record.provenance.fixture_search_counter.is_some() {
                return Err(invalid(
                    "representative record carries a fixture search counter",
                ));
            }
            let expected = seed_commitment(derive_seed(GAME_SEED_DOMAIN, &[game_index]));
            if record.provenance.seed_commitment_blake3 != expected {
                return Err(invalid("representative seed commitment drifted"));
            }
            if usize::from(record.state.focal_seat) >= PLAYERS
                || record.provenance.policy
                    != policy_for(game_index, usize::from(record.state.focal_seat))
            {
                return Err(invalid(
                    "representative trajectory policy provenance drifted",
                ));
            }
        }
        RecordKind::ReachableOverflowWitness => {
            let counter = record
                .provenance
                .fixture_search_counter
                .ok_or_else(|| invalid("overflow witness has no search counter"))?;
            if record.provenance.game_index.is_some()
                || record.provenance.policy != TrajectoryPolicy::Greedy
                || counter >= OVERFLOW_WITNESS_MAX_SEARCH
            {
                return Err(invalid("overflow witness provenance is malformed"));
            }
            let expected = seed_commitment(derive_seed(OVERFLOW_WITNESS_DOMAIN, &[counter]));
            if record.provenance.seed_commitment_blake3 != expected {
                return Err(invalid("overflow witness seed commitment drifted"));
            }
        }
    }
    Ok(())
}

fn validate_public_state_shape(record: &PublicStateRecord) -> Result<()> {
    let state = &record.state;
    if state.ruleset != "four_player_aaaaa_no_habitat_bonus_labels"
        || state.num_players != PLAYERS as u8
        || state.boards.len() != PLAYERS
        || usize::from(state.focal_seat) >= PLAYERS
        || state.free_overflow_applied != state.overflow_used_this_turn
    {
        return Err(invalid("public state top-level contract drifted"));
    }
    let decision = usize::from(record.provenance.decision_index);
    if decision >= ROWS_PER_GAME
        || usize::from(state.focal_seat) != decision % PLAYERS
        || state.personal_turn != u8::try_from(decision / PLAYERS + 1)?
        || state.phase != phase_for_turn(state.personal_turn)?
        || state.turns_remaining != u8::try_from(ROWS_PER_GAME - decision)?
    {
        return Err(invalid(
            "public state turn, seat, or phase metadata drifted",
        ));
    }
    if state.market.iter().any(Option::is_none) {
        return Err(invalid(
            "public state market must contain four visible pairs",
        ));
    }

    let mut drafted_tiles = 0usize;
    for (seat, board) in state.boards.iter().enumerate() {
        if usize::from(board.absolute_seat) != seat {
            return Err(invalid("public board absolute seat order drifted"));
        }
        let expected_completed = completed_turns_before(decision, seat);
        if board.tiles.len() != 3 + expected_completed {
            return Err(invalid(format!(
                "public board {seat} has {} tiles, expected {}",
                board.tiles.len(),
                3 + expected_completed
            )));
        }
        for (ordinal, tile) in board.tiles.iter().enumerate() {
            if usize::from(tile.placement_ordinal) != ordinal
                || tile.starter != (ordinal < 3)
                || HexCoord::new(tile.q, tile.r).to_index().is_none()
            {
                return Err(invalid("public board tile ordering or coordinate drifted"));
            }
            let _ = tile.tile_data()?;
            if let Some(wildlife) = tile.wildlife
                && Wildlife::from_u8(wildlife).is_none()
            {
                return Err(invalid("public board wildlife is out of range"));
            }
            if !tile.starter {
                drafted_tiles += 1;
            }
        }
    }
    if drafted_tiles != decision {
        return Err(invalid("public state drafted tile conservation failed"));
    }
    let projection = derive_public_supply(&state.boards, &state.market)?;
    if projection.supply != state.supply
        || usize::from(state.supply.remaining_tiles) != 81usize.saturating_sub(decision)
    {
        return Err(invalid("public state supply conservation failed"));
    }
    Ok(())
}

fn reconstruct_extractor_inputs(state: &PublicState) -> Result<(Board, BagInfo)> {
    let boards = state
        .boards
        .iter()
        .map(reconstruct_board)
        .collect::<Result<Vec<_>>>()?;
    let pairs = state
        .market
        .map(|slot| slot.map(PublicMarketPair::market_pair).transpose())
        .into_iter()
        .collect::<Result<Vec<_>>>()?;
    let pairs: [Option<MarketPair>; 4] = pairs
        .try_into()
        .map_err(|_| invalid("public market width drifted"))?;
    let projection = derive_public_supply(&state.boards, &state.market)?;

    let mut game = template_game().clone();
    game.boards = boards.clone();
    game.market = Market { pairs };
    game.current_player = usize::from(state.focal_seat);
    game.turns_remaining = state.turns_remaining;
    game.num_players = PLAYERS;
    game.overflow_used_this_turn = state.overflow_used_this_turn;
    let mut bag = BagInfo::from_game_for_player(&game, usize::from(state.focal_seat));
    bag.tbag_terrain = projection.supply.terrain_counts;
    bag.tbag_wildlife = projection.supply.wildlife_capacity_counts;
    bag.tbag_joint = projection.tbag_joint;
    if bag.remaining != projection.supply.wildlife_remaining {
        return Err(invalid(
            "reconstructed BagInfo wildlife remaining disagrees with public state",
        ));
    }
    Ok((boards[usize::from(state.focal_seat)].clone(), bag))
}

fn reconstruct_board(public: &PublicBoard) -> Result<Board> {
    let mut board = Board::new();
    for tile in &public.tiles {
        board
            .place_tile(
                HexCoord::new(tile.q, tile.r),
                tile.tile_data()?,
                tile.rotation,
            )
            .ok_or_else(|| invalid("public board tile reconstruction failed"))?;
    }
    for tile in &public.tiles {
        if let Some(wildlife) = tile.wildlife {
            let index = HexCoord::new(tile.q, tile.r)
                .to_index()
                .ok_or_else(|| invalid("public wildlife coordinate is out of range"))?;
            board
                .place_wildlife(
                    index,
                    Wildlife::from_u8(wildlife)
                        .ok_or_else(|| invalid("public wildlife is out of range"))?,
                )
                .ok_or_else(|| invalid("public wildlife reconstruction failed"))?;
        }
    }
    board.nature_tokens = public.nature_tokens;
    Ok(board)
}

fn derive_public_supply(
    boards: &[PublicBoard],
    market: &[Option<PublicMarketPair>; 4],
) -> Result<PublicSupplyProjection> {
    let mut inventory = standard_tile_inventory().clone();
    for board in boards {
        for tile in &board.tiles {
            if tile.starter {
                continue;
            }
            subtract_tile(&mut inventory, TileKey::from_public(tile)?)?;
        }
    }
    for pair in market.iter().flatten() {
        subtract_tile(&mut inventory, TileKey::from_market(*pair)?)?;
    }

    let mut terrain_counts = [0u8; 5];
    let mut wildlife_capacity_counts = [0u8; 5];
    let mut tbag_joint = [[0u8; 5]; 5];
    let mut remaining_tiles = 0u16;
    for (tile, count) in inventory {
        remaining_tiles = remaining_tiles
            .checked_add(count)
            .ok_or_else(|| invalid("remaining tile count overflowed"))?;
        let t1 = usize::from(tile.terrain1);
        terrain_counts[t1] = checked_add_u8(terrain_counts[t1], count)?;
        if let Some(t2) = tile.terrain2 {
            terrain_counts[usize::from(t2)] =
                checked_add_u8(terrain_counts[usize::from(t2)], count)?;
        }
        for wildlife in 0..5 {
            if tile.allowed_mask & (1 << wildlife) == 0 {
                continue;
            }
            wildlife_capacity_counts[wildlife] =
                checked_add_u8(wildlife_capacity_counts[wildlife], count)?;
            tbag_joint[t1][wildlife] = checked_add_u8(tbag_joint[t1][wildlife], count)?;
            if let Some(t2) = tile.terrain2 {
                tbag_joint[usize::from(t2)][wildlife] =
                    checked_add_u8(tbag_joint[usize::from(t2)][wildlife], count)?;
            }
        }
    }

    let mut visible_wildlife = [0u8; 5];
    for board in boards {
        for tile in &board.tiles {
            if let Some(wildlife) = tile.wildlife {
                let slot = visible_wildlife
                    .get_mut(usize::from(wildlife))
                    .ok_or_else(|| invalid("public board wildlife is out of range"))?;
                *slot = slot
                    .checked_add(1)
                    .ok_or_else(|| invalid("visible wildlife count overflowed"))?;
            }
        }
    }
    for pair in market.iter().flatten() {
        let slot = visible_wildlife
            .get_mut(usize::from(pair.wildlife))
            .ok_or_else(|| invalid("public market wildlife is out of range"))?;
        *slot = slot
            .checked_add(1)
            .ok_or_else(|| invalid("visible wildlife count overflowed"))?;
    }
    let mut wildlife_remaining = [0u8; 5];
    for wildlife in 0..5 {
        if visible_wildlife[wildlife] > 20 {
            return Err(invalid("public wildlife conservation exceeds 20 tokens"));
        }
        wildlife_remaining[wildlife] = 20 - visible_wildlife[wildlife];
    }

    Ok(PublicSupplyProjection {
        supply: PublicSupply {
            remaining_tiles: u8::try_from(remaining_tiles)?,
            terrain_counts,
            wildlife_capacity_counts,
            wildlife_remaining,
        },
        tbag_joint,
    })
}

fn subtract_tile(inventory: &mut BTreeMap<TileKey, u16>, tile: TileKey) -> Result<()> {
    let count = inventory
        .get_mut(&tile)
        .ok_or_else(|| invalid("public drafted tile is absent from standard tile inventory"))?;
    if *count == 0 {
        return Err(invalid(
            "public drafted tile multiplicity exceeds standard inventory",
        ));
    }
    *count -= 1;
    Ok(())
}

fn checked_add_u8(current: u8, value: u16) -> Result<u8> {
    let total = u16::from(current)
        .checked_add(value)
        .ok_or_else(|| invalid("public supply count overflowed"))?;
    Ok(u8::try_from(total)?)
}

fn standard_tile_inventory() -> &'static BTreeMap<TileKey, u16> {
    static INVENTORY: OnceLock<BTreeMap<TileKey, u16>> = OnceLock::new();
    INVENTORY.get_or_init(|| {
        let mut rng = StdRng::seed_from_u64(0);
        let mut bag = TileBag::new(&mut rng);
        let mut inventory = BTreeMap::new();
        while let Some(tile) = bag.draw() {
            *inventory.entry(TileKey::from_tile(tile)).or_default() += 1;
        }
        assert_eq!(inventory.values().copied().sum::<u16>(), 85);
        inventory
    })
}

fn template_game() -> &'static GameState {
    static TEMPLATE: OnceLock<GameState> = OnceLock::new();
    TEMPLATE.get_or_init(|| {
        let mut rng = StdRng::seed_from_u64(0);
        GameState::new(PLAYERS, ScoringCards::all_a(), &mut rng)
    })
}

fn normalized_features(board: &Board, bag: &BagInfo) -> Result<(usize, Vec<u16>)> {
    let mut features = extract_features_with_bag(board, Some(bag));
    let raw = features.len();
    features.sort_unstable();
    features.dedup();
    if features.is_empty()
        || features
            .last()
            .is_some_and(|feature| usize::from(*feature) >= NUM_FEATURES)
    {
        return Err(invalid(
            "actual corrected Rust extractor emitted an invalid feature stream",
        ));
    }
    Ok((raw, features))
}

fn corrected_tail_from(features: &[u16]) -> Result<Vec<u16>> {
    let tail = features
        .iter()
        .copied()
        .filter(|feature| {
            (LEGACY_MID_V4_FIXED_V1_TBAG_TERRAIN_BASE..NUM_FEATURES_LEGACY_MID_V4_FIXED_V1)
                .contains(&usize::from(*feature))
        })
        .collect::<Vec<_>>();
    if tail.windows(2).any(|window| window[0] >= window[1]) {
        return Err(invalid("corrected tail features are not strictly ordered"));
    }
    Ok(tail)
}

fn validate_corrected_tail_shape(features: &[u16], overflow_used: bool) -> Result<()> {
    let terrain = features
        .iter()
        .filter(|feature| {
            (LEGACY_MID_V4_FIXED_V1_TBAG_TERRAIN_BASE..LEGACY_MID_V4_FIXED_V1_TBAG_WL_BASE)
                .contains(&usize::from(**feature))
        })
        .count();
    let wildlife = features
        .iter()
        .filter(|feature| {
            (LEGACY_MID_V4_FIXED_V1_TBAG_WL_BASE..LEGACY_MID_V4_FIXED_V1_OVERFLOW_BASE)
                .contains(&usize::from(**feature))
        })
        .count();
    let overflow = features.contains(&(LEGACY_MID_V4_FIXED_V1_OVERFLOW_BASE as u16));
    if terrain != 5 || wildlife != 5 || overflow != overflow_used {
        return Err(invalid(format!(
            "corrected tail shape drifted: terrain={terrain}, wildlife={wildlife}, \
             overflow={overflow}, expected_overflow={overflow_used}"
        )));
    }
    Ok(())
}

pub fn corrected_tail_indices() -> Vec<u16> {
    (LEGACY_MID_V4_FIXED_V1_TBAG_TERRAIN_BASE..NUM_FEATURES_LEGACY_MID_V4_FIXED_V1)
        .map(|index| index as u16)
        .collect()
}

pub(crate) fn hash_u16_features(domain: &[u8], features: &[u16]) -> String {
    let mut hasher = Hasher::new();
    hasher.update(domain);
    for feature in features {
        update_framed(&mut hasher, &feature.to_le_bytes());
    }
    hasher.finalize().to_hex().to_string()
}

fn derive_seed(domain: &[u8], values: &[u64]) -> u64 {
    let mut hasher = Hasher::new();
    hasher.update(domain);
    for value in values {
        hasher.update(&value.to_le_bytes());
    }
    u64::from_le_bytes(
        hasher.finalize().as_bytes()[..8]
            .try_into()
            .expect("BLAKE3 digest has eight bytes"),
    )
}

fn seed_commitment(seed: u64) -> String {
    let mut hasher = Hasher::new();
    hasher.update(b"corrected-mid-tail-activation-census-v1/seed-commitment/v1");
    hasher.update(&seed.to_le_bytes());
    hasher.finalize().to_hex().to_string()
}

fn completed_turns_before(decision: usize, seat: usize) -> usize {
    if decision <= seat {
        0
    } else {
        (decision - 1 - seat) / PLAYERS + 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compiled_layout_is_the_corrected_schema() {
        validate_compiled_contract().unwrap();
        assert_eq!(corrected_tail_indices().len(), 301);
        assert_eq!(corrected_tail_indices()[0], 10_930);
        assert_eq!(corrected_tail_indices()[300], 11_230);
    }

    #[test]
    fn phase_boundaries_are_frozen() {
        assert_eq!(phase_for_turn(1).unwrap(), Phase::Opening);
        assert_eq!(phase_for_turn(2).unwrap(), Phase::Early);
        assert_eq!(phase_for_turn(5).unwrap(), Phase::Early);
        assert_eq!(phase_for_turn(6).unwrap(), Phase::Middle);
        assert_eq!(phase_for_turn(13).unwrap(), Phase::Middle);
        assert_eq!(phase_for_turn(14).unwrap(), Phase::Late);
        assert_eq!(phase_for_turn(20).unwrap(), Phase::Late);
        assert!(phase_for_turn(0).is_err());
        assert!(phase_for_turn(21).is_err());
    }

    #[test]
    fn modulo_ownership_is_disjoint_and_complete() {
        let mut all = Vec::new();
        for shard in 0..4 {
            let owned = owned_game_indices(100, 17, shard, 4).unwrap();
            assert!(owned.iter().all(|index| (*index - 100) % 4 == shard as u64));
            all.extend(owned);
        }
        all.sort_unstable();
        assert_eq!(all, (100..117).collect::<Vec<_>>());
    }

    #[test]
    fn representative_record_round_trips_through_public_state_only() {
        let records = generate_representative_game(7).unwrap();
        assert_eq!(records.len(), ROWS_PER_GAME);
        for record in [&records[0], &records[37], &records[79]] {
            let state = serde_json::to_value(&record.state).unwrap();
            let provenance = serde_json::to_value(&record.provenance).unwrap();
            assert!(state.get("policy").is_none());
            assert_eq!(
                provenance["policy"],
                serde_json::to_value(record.provenance.policy).unwrap()
            );
            assert_eq!(
                record.public_state_blake3,
                canonical_blake3(&record.state).unwrap()
            );
            let replayed = replay_record(record).unwrap();
            assert_eq!(
                replayed.corrected_tail_features,
                record.corrected_tail_features
            );
        }
    }

    #[test]
    fn reachable_overflow_witness_activates_row_11230() {
        let witness = generate_reachable_overflow_witness().unwrap();
        assert!(witness.state.overflow_used_this_turn);
        assert!(
            witness
                .corrected_tail_features
                .contains(&(LEGACY_MID_V4_FIXED_V1_OVERFLOW_BASE as u16))
        );
        replay_record(&witness).unwrap();
    }

    #[test]
    fn corrected_count_blocks_emit_five_rows_each() {
        let record = &generate_representative_game(11).unwrap()[0];
        let terrain = record
            .corrected_tail_features
            .iter()
            .filter(|feature| (10_930..11_080).contains(&usize::from(**feature)))
            .count();
        let wildlife = record
            .corrected_tail_features
            .iter()
            .filter(|feature| (11_080..11_230).contains(&usize::from(**feature)))
            .count();
        assert_eq!(terrain, 5);
        assert_eq!(wildlife, 5);
    }
}
