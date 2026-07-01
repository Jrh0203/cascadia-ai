use std::collections::{BTreeMap, BTreeSet};

use r2_sparse_entity_census::{MarketToken, SuppliedTile};
use serde::{Deserialize, Serialize};

use crate::model::FarFrontierSignature;
use crate::{
    AdaptiveMultiResolutionState, FarFrontierBucket, FarHabitatComponent, FarWildlifeComponent,
    FarWildlifeMotifBucket, NearCell, NearCellState, NearFieldRadius, R4Error, Result,
};

pub const BOUNDED_MAGIC: &[u8; 8] = b"CSR4BQ1\0";
pub const BOUNDED_SCHEMA_VERSION: u16 = 1;
pub const BOUNDED_MAX_TOKENS: usize = 224;
pub const BOUNDED_P99_TOKEN_LIMIT: u64 = 192;
pub const BOUNDED_ACTIVE_SCALAR_LIMIT: u64 = 16_384;
pub const BOUNDED_PADDED_SCALAR_LIMIT: u64 = 24_576;
pub const BOUNDED_BYTE_LIMIT: u64 = 65_536;
pub const BOUNDED_THROUGHPUT_RATIO_MINIMUM: f64 = 0.90;

const RELATIVE_SEATS: usize = 4;
const WILDLIFE_TYPES: usize = 5;
const TERRAIN_TYPES: usize = 5;
const HEX_SECTORS: usize = 6;
const RADIAL_BINS: usize = 16;
const SECTOR_CLASSES: usize = 13;
const NEAR_BITMAP_WORDS: usize = 4;
const NONE: u8 = u8::MAX;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
#[repr(u8)]
pub enum BoundedArm {
    SeatMarginal = 1,
    Directional = 2,
    Affordance = 3,
    SelectiveExact = 4,
}

impl BoundedArm {
    pub const ALL: [Self; 4] = [
        Self::SeatMarginal,
        Self::Directional,
        Self::Affordance,
        Self::SelectiveExact,
    ];

    pub const fn id(self) -> &'static str {
        match self {
            Self::SeatMarginal => "q1-seat-marginal",
            Self::Directional => "q2-directional",
            Self::Affordance => "q3-affordance",
            Self::SelectiveExact => "q4-selective-exact",
        }
    }

    pub const fn code(self) -> u8 {
        self as u8
    }

    pub const fn from_code(code: u8) -> Option<Self> {
        match code {
            1 => Some(Self::SeatMarginal),
            2 => Some(Self::Directional),
            3 => Some(Self::Affordance),
            4 => Some(Self::SelectiveExact),
            _ => None,
        }
    }

    pub fn from_id(id: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|arm| arm.id() == id)
    }

    pub const fn hard_token_max(self) -> usize {
        match self {
            Self::SeatMarginal => 184,
            Self::Directional => 204,
            Self::Affordance => 200,
            Self::SelectiveExact => 224,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
#[repr(u8)]
pub enum BoundedTokenKind {
    NearCell = 1,
    HabitatComponent = 2,
    WildlifeComponent = 3,
    WildlifeSummary = 4,
    FrontierSummary = 5,
    ExactWildlifeBucket = 6,
    ExactFrontierBucket = 7,
}

impl BoundedTokenKind {
    pub const fn code(self) -> u8 {
        self as u8
    }

    pub const fn from_code(code: u8) -> Option<Self> {
        match code {
            1 => Some(Self::NearCell),
            2 => Some(Self::HabitatComponent),
            3 => Some(Self::WildlifeComponent),
            4 => Some(Self::WildlifeSummary),
            5 => Some(Self::FrontierSummary),
            6 => Some(Self::ExactWildlifeBucket),
            7 => Some(Self::ExactFrontierBucket),
            _ => None,
        }
    }

    pub const fn padded_width(self) -> usize {
        match self {
            Self::NearCell => 128,
            Self::HabitatComponent => 64,
            Self::WildlifeComponent => 80,
            Self::WildlifeSummary => 80,
            Self::FrontierSummary => 144,
            Self::ExactWildlifeBucket => 16,
            Self::ExactFrontierBucket => 80,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BoundedToken {
    pub kind: BoundedTokenKind,
    pub values: Vec<i16>,
}

impl BoundedToken {
    fn new(kind: BoundedTokenKind, values: Vec<i16>) -> Result<Self> {
        if values.len() > kind.padded_width() {
            return Err(R4Error::InvalidBoundedView(format!(
                "{} token has {} active scalars; padded width is {}",
                kind.code(),
                values.len(),
                kind.padded_width()
            )));
        }
        Ok(Self { kind, values })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BoundedGlobalView {
    pub turn: u8,
    pub current_relative_seat: u8,
    pub player_count: u8,
    pub total_turns: u8,
    pub scoring_cards: [u8; 5],
    pub habitat_bonuses: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BoundedPlayerView {
    pub relative_seat: u8,
    pub turns_taken: u8,
    pub turns_until_next_action: u8,
    pub occupied_count: u8,
    pub nature_tokens: u8,
    pub wildlife_counts: [u8; 5],
    pub largest_habitats: [u8; 5],
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BoundedMarketView {
    pub slot: u8,
    pub tile: Option<[u8; 4]>,
    pub wildlife: Option<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BoundedAccounting {
    pub source_wildlife_buckets: u32,
    pub source_wildlife_mass: u32,
    pub summarized_wildlife_buckets: u32,
    pub summarized_wildlife_mass: u32,
    pub exact_wildlife_buckets: u32,
    pub exact_wildlife_mass: u32,
    pub source_frontier_buckets: u32,
    pub source_frontier_mass: u32,
    pub summarized_frontier_buckets: u32,
    pub summarized_frontier_mass: u32,
    pub exact_frontier_buckets: u32,
    pub exact_frontier_mass: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BoundedFeatureView {
    pub schema_version: u16,
    pub arm: BoundedArm,
    pub global: BoundedGlobalView,
    pub players: Vec<BoundedPlayerView>,
    pub market: Vec<BoundedMarketView>,
    pub supplied_tile: Option<[u8; 4]>,
    pub accounting: BoundedAccounting,
    pub tokens: Vec<BoundedToken>,
}

impl BoundedFeatureView {
    pub fn from_state(state: &AdaptiveMultiResolutionState, arm: BoundedArm) -> Result<Self> {
        if state.radius != NearFieldRadius::Radius4 {
            return Err(R4Error::InvalidBoundedView(
                "bounded quotient requires radius4-61".to_owned(),
            ));
        }
        let focal = state
            .boards
            .get(usize::from(state.focal_relative_seat))
            .ok_or_else(|| R4Error::InvalidBoundedView("focal board is missing".to_owned()))?;

        let wildlife_buckets = state
            .boards
            .iter()
            .flat_map(|board| board.far_wildlife_motif_buckets.iter())
            .cloned()
            .collect::<Vec<_>>();
        let frontier_buckets = state
            .boards
            .iter()
            .flat_map(|board| board.far_frontier_buckets.iter())
            .cloned()
            .collect::<Vec<_>>();

        let selected_wildlife = if arm == BoundedArm::SelectiveExact {
            select_wildlife_indices(&wildlife_buckets, 16)?
        } else {
            BTreeSet::new()
        };
        let selected_frontier = if arm == BoundedArm::SelectiveExact {
            select_frontier_indices(&frontier_buckets, 24)?
        } else {
            BTreeSet::new()
        };

        let residual_wildlife = wildlife_buckets
            .iter()
            .enumerate()
            .filter(|(index, _)| !selected_wildlife.contains(index))
            .map(|(_, bucket)| bucket)
            .collect::<Vec<_>>();
        let residual_frontier = frontier_buckets
            .iter()
            .enumerate()
            .filter(|(index, _)| !selected_frontier.contains(index))
            .map(|(_, bucket)| bucket)
            .collect::<Vec<_>>();

        let mut tokens = Vec::with_capacity(arm.hard_token_max());
        for cell in &focal.near_cells {
            tokens.push(near_token(cell)?);
        }
        for component in state
            .boards
            .iter()
            .flat_map(|board| board.far_habitat_components.iter())
        {
            tokens.push(habitat_component_token(component)?);
        }
        for component in state
            .boards
            .iter()
            .flat_map(|board| board.far_wildlife_components.iter())
        {
            tokens.push(wildlife_component_token(component)?);
        }
        tokens.extend(wildlife_summary_tokens(&residual_wildlife)?);
        tokens.extend(frontier_summary_tokens(arm, &residual_frontier)?);

        if arm == BoundedArm::SelectiveExact {
            for index in &selected_wildlife {
                tokens.push(exact_wildlife_bucket_token(&wildlife_buckets[*index])?);
            }
            for index in &selected_frontier {
                tokens.push(exact_frontier_bucket_token(&frontier_buckets[*index])?);
            }
        }

        let view = Self {
            schema_version: BOUNDED_SCHEMA_VERSION,
            arm,
            global: BoundedGlobalView {
                turn: state.global.turn,
                current_relative_seat: state.global.current_relative_seat,
                player_count: state.global.player_count,
                total_turns: state.global.total_turns,
                scoring_cards: state.global.scoring_cards,
                habitat_bonuses: state.global.habitat_bonuses,
            },
            players: state
                .players
                .iter()
                .map(|player| BoundedPlayerView {
                    relative_seat: player.relative_seat,
                    turns_taken: player.turns_taken,
                    turns_until_next_action: player.turns_until_next_action,
                    occupied_count: player.occupied_count,
                    nature_tokens: player.nature_tokens,
                    wildlife_counts: player.wildlife_counts,
                    largest_habitats: player.largest_habitats,
                })
                .collect(),
            market: state.market.iter().map(market_view).collect(),
            supplied_tile: state.supplied_tile.map(tile_view),
            accounting: BoundedAccounting {
                source_wildlife_buckets: checked_len_u32(
                    wildlife_buckets.len(),
                    "source wildlife bucket count",
                )?,
                source_wildlife_mass: checked_count_sum(
                    wildlife_buckets.iter().map(|bucket| bucket.count),
                    "source wildlife mass",
                )?,
                summarized_wildlife_buckets: checked_len_u32(
                    residual_wildlife.len(),
                    "summarized wildlife bucket count",
                )?,
                summarized_wildlife_mass: checked_count_sum(
                    residual_wildlife.iter().map(|bucket| bucket.count),
                    "summarized wildlife mass",
                )?,
                exact_wildlife_buckets: checked_len_u32(
                    selected_wildlife.len(),
                    "exact wildlife bucket count",
                )?,
                exact_wildlife_mass: checked_count_sum(
                    selected_wildlife
                        .iter()
                        .map(|index| wildlife_buckets[*index].count),
                    "exact wildlife mass",
                )?,
                source_frontier_buckets: checked_len_u32(
                    frontier_buckets.len(),
                    "source frontier bucket count",
                )?,
                source_frontier_mass: checked_count_sum(
                    frontier_buckets.iter().map(|bucket| bucket.count),
                    "source frontier mass",
                )?,
                summarized_frontier_buckets: checked_len_u32(
                    residual_frontier.len(),
                    "summarized frontier bucket count",
                )?,
                summarized_frontier_mass: checked_count_sum(
                    residual_frontier.iter().map(|bucket| bucket.count),
                    "summarized frontier mass",
                )?,
                exact_frontier_buckets: checked_len_u32(
                    selected_frontier.len(),
                    "exact frontier bucket count",
                )?,
                exact_frontier_mass: checked_count_sum(
                    selected_frontier
                        .iter()
                        .map(|index| frontier_buckets[*index].count),
                    "exact frontier mass",
                )?,
            },
            tokens,
        };
        view.validate()?;
        Ok(view)
    }

    pub fn validate(&self) -> Result<()> {
        if self.schema_version != BOUNDED_SCHEMA_VERSION {
            return Err(R4Error::InvalidBoundedView(format!(
                "unsupported bounded schema {}",
                self.schema_version
            )));
        }
        if self.players.len() != usize::from(self.global.player_count)
            || self.global.player_count == 0
            || usize::from(self.global.player_count) > RELATIVE_SEATS
            || self.global.current_relative_seat >= self.global.player_count
            || self.global.turn > self.global.total_turns
            || self.global.scoring_cards.iter().any(|card| *card > 3)
        {
            return Err(R4Error::InvalidBoundedView(
                "bounded player metadata is inconsistent".to_owned(),
            ));
        }
        for (relative_seat, player) in self.players.iter().enumerate() {
            if usize::from(player.relative_seat) != relative_seat {
                return Err(R4Error::InvalidBoundedView(
                    "bounded players are not in canonical relative-seat order".to_owned(),
                ));
            }
        }
        validate_market(&self.market)?;
        if let Some(tile) = self.supplied_tile {
            validate_tile(tile, "supplied tile")?;
        }
        if self.tokens.len() > self.arm.hard_token_max() || self.tokens.len() > BOUNDED_MAX_TOKENS {
            return Err(R4Error::InvalidBoundedView(format!(
                "{} emits {} tokens above its hard maximum {}",
                self.arm.id(),
                self.tokens.len(),
                self.arm.hard_token_max()
            )));
        }
        if self
            .tokens
            .windows(2)
            .any(|pair| pair[0].kind > pair[1].kind)
        {
            return Err(R4Error::InvalidBoundedView(
                "bounded token kinds are not in canonical order".to_owned(),
            ));
        }
        for token in &self.tokens {
            if token.values.len() > token.kind.padded_width() {
                return Err(R4Error::InvalidBoundedView(
                    "bounded token exceeds its type width".to_owned(),
                ));
            }
            validate_token_shape(token)?;
        }
        let near_count = self
            .tokens
            .iter()
            .filter(|token| token.kind == BoundedTokenKind::NearCell)
            .count();
        if near_count != NearFieldRadius::Radius4.capacity() {
            return Err(R4Error::InvalidBoundedView(format!(
                "bounded view contains {near_count} near cells; expected {}",
                NearFieldRadius::Radius4.capacity()
            )));
        }
        let wildlife_summary_count = self
            .tokens
            .iter()
            .filter(|token| token.kind == BoundedTokenKind::WildlifeSummary)
            .count();
        let frontier_summary_count = self
            .tokens
            .iter()
            .filter(|token| token.kind == BoundedTokenKind::FrontierSummary)
            .count();
        let expected_frontier = match self.arm {
            BoundedArm::SeatMarginal | BoundedArm::SelectiveExact => RELATIVE_SEATS,
            BoundedArm::Directional => RELATIVE_SEATS * HEX_SECTORS,
            BoundedArm::Affordance => RELATIVE_SEATS * TERRAIN_TYPES,
        };
        if wildlife_summary_count != RELATIVE_SEATS * WILDLIFE_TYPES
            || frontier_summary_count != expected_frontier
        {
            return Err(R4Error::InvalidBoundedView(
                "bounded summary token grid is incomplete".to_owned(),
            ));
        }
        validate_summary_grids(self)?;
        if self.accounting.source_wildlife_buckets
            != checked_pair_sum(
                self.accounting.summarized_wildlife_buckets,
                self.accounting.exact_wildlife_buckets,
                "wildlife bucket accounting",
            )?
            || self.accounting.source_wildlife_mass
                != checked_pair_sum(
                    self.accounting.summarized_wildlife_mass,
                    self.accounting.exact_wildlife_mass,
                    "wildlife mass accounting",
                )?
            || self.accounting.source_frontier_buckets
                != checked_pair_sum(
                    self.accounting.summarized_frontier_buckets,
                    self.accounting.exact_frontier_buckets,
                    "frontier bucket accounting",
                )?
            || self.accounting.source_frontier_mass
                != checked_pair_sum(
                    self.accounting.summarized_frontier_mass,
                    self.accounting.exact_frontier_mass,
                    "frontier mass accounting",
                )?
        {
            return Err(R4Error::InvalidBoundedView(
                "bounded exact-plus-summary accounting is not lossless".to_owned(),
            ));
        }
        if self.arm != BoundedArm::SelectiveExact
            && (self.accounting.exact_wildlife_buckets != 0
                || self.accounting.exact_frontier_buckets != 0)
        {
            return Err(R4Error::InvalidBoundedView(
                "nonselective arm contains exact bucket accounting".to_owned(),
            ));
        }
        if self.arm == BoundedArm::SelectiveExact
            && (self.accounting.exact_wildlife_buckets > 16
                || self.accounting.exact_frontier_buckets > 24)
        {
            return Err(R4Error::InvalidBoundedView(
                "selective exact arm exceeds its exact-bucket limits".to_owned(),
            ));
        }
        Ok(())
    }

    pub fn spatial_token_count(&self) -> usize {
        self.tokens.len()
    }

    pub fn active_scalar_count(&self) -> usize {
        metadata_scalar_count(self)
            + self
                .tokens
                .iter()
                .map(|token| token.values.len())
                .sum::<usize>()
    }

    pub fn padded_scalar_slots(&self) -> usize {
        metadata_scalar_count(self)
            + self
                .tokens
                .iter()
                .map(|token| token.kind.padded_width())
                .sum::<usize>()
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>> {
        self.validate()?;
        let mut writer = EnvelopeWriter::default();
        writer.raw(BOUNDED_MAGIC);
        writer.u16(self.schema_version);
        writer.u8(self.arm.code());
        writer.u8(self.global.turn);
        writer.u8(self.global.current_relative_seat);
        writer.u8(self.global.player_count);
        writer.u8(self.global.total_turns);
        writer.raw(&self.global.scoring_cards);
        writer.u8(u8::from(self.global.habitat_bonuses));
        writer.u8(checked_u8(self.players.len(), "player count")?);
        for player in &self.players {
            writer.u8(player.relative_seat);
            writer.u8(player.turns_taken);
            writer.u8(player.turns_until_next_action);
            writer.u8(player.occupied_count);
            writer.u8(player.nature_tokens);
            writer.raw(&player.wildlife_counts);
            writer.raw(&player.largest_habitats);
        }
        writer.u8(checked_u8(self.market.len(), "market count")?);
        for market in &self.market {
            writer.u8(market.slot);
            writer.u8(u8::from(market.tile.is_some()));
            if let Some(tile) = market.tile {
                writer.raw(&tile);
            }
            writer.u8(market.wildlife.unwrap_or(NONE));
        }
        writer.u8(u8::from(self.supplied_tile.is_some()));
        if let Some(tile) = self.supplied_tile {
            writer.raw(&tile);
        }
        for value in accounting_values(&self.accounting) {
            writer.u32(value);
        }
        writer.u16(checked_u16(self.tokens.len(), "token count")?);
        for token in &self.tokens {
            writer.u8(token.kind.code());
            writer.u16(checked_u16(token.values.len(), "active token width")?);
            writer.u16(checked_u16(
                token.kind.padded_width(),
                "padded token width",
            )?);
            for value in &token.values {
                writer.i16(*value);
            }
            for _ in token.values.len()..token.kind.padded_width() {
                writer.i16(0);
            }
        }
        Ok(writer.bytes)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self> {
        let mut reader = EnvelopeReader::new(bytes);
        if reader.raw(BOUNDED_MAGIC.len())? != BOUNDED_MAGIC {
            return Err(R4Error::InvalidBoundedEnvelope(
                "bounded magic mismatch".to_owned(),
            ));
        }
        let schema_version = reader.u16()?;
        let arm = BoundedArm::from_code(reader.u8()?).ok_or_else(|| {
            R4Error::InvalidBoundedEnvelope("bounded arm code is invalid".to_owned())
        })?;
        let global = BoundedGlobalView {
            turn: reader.u8()?,
            current_relative_seat: reader.u8()?,
            player_count: reader.u8()?,
            total_turns: reader.u8()?,
            scoring_cards: reader.array_u8::<5>()?,
            habitat_bonuses: read_bool(&mut reader)?,
        };
        let player_count = usize::from(reader.u8()?);
        let mut players = Vec::with_capacity(player_count);
        for _ in 0..player_count {
            players.push(BoundedPlayerView {
                relative_seat: reader.u8()?,
                turns_taken: reader.u8()?,
                turns_until_next_action: reader.u8()?,
                occupied_count: reader.u8()?,
                nature_tokens: reader.u8()?,
                wildlife_counts: reader.array_u8::<5>()?,
                largest_habitats: reader.array_u8::<5>()?,
            });
        }
        let market_count = usize::from(reader.u8()?);
        let mut market = Vec::with_capacity(market_count);
        for _ in 0..market_count {
            let slot = reader.u8()?;
            let tile = read_bool(&mut reader)?
                .then(|| reader.array_u8::<4>())
                .transpose()?;
            let wildlife = match reader.u8()? {
                NONE => None,
                value if usize::from(value) < WILDLIFE_TYPES => Some(value),
                _ => {
                    return Err(R4Error::InvalidBoundedEnvelope(
                        "market wildlife code is invalid".to_owned(),
                    ));
                }
            };
            market.push(BoundedMarketView {
                slot,
                tile,
                wildlife,
            });
        }
        let supplied_tile = read_bool(&mut reader)?
            .then(|| reader.array_u8::<4>())
            .transpose()?;
        let mut accounting_array = [0u32; 12];
        for value in &mut accounting_array {
            *value = reader.u32()?;
        }
        let accounting = BoundedAccounting {
            source_wildlife_buckets: accounting_array[0],
            source_wildlife_mass: accounting_array[1],
            summarized_wildlife_buckets: accounting_array[2],
            summarized_wildlife_mass: accounting_array[3],
            exact_wildlife_buckets: accounting_array[4],
            exact_wildlife_mass: accounting_array[5],
            source_frontier_buckets: accounting_array[6],
            source_frontier_mass: accounting_array[7],
            summarized_frontier_buckets: accounting_array[8],
            summarized_frontier_mass: accounting_array[9],
            exact_frontier_buckets: accounting_array[10],
            exact_frontier_mass: accounting_array[11],
        };
        let token_count = usize::from(reader.u16()?);
        let mut tokens = Vec::with_capacity(token_count);
        for _ in 0..token_count {
            let kind = BoundedTokenKind::from_code(reader.u8()?).ok_or_else(|| {
                R4Error::InvalidBoundedEnvelope("bounded token kind is invalid".to_owned())
            })?;
            let active = usize::from(reader.u16()?);
            let padded = usize::from(reader.u16()?);
            if padded != kind.padded_width() || active > padded {
                return Err(R4Error::InvalidBoundedEnvelope(
                    "bounded token width is noncanonical".to_owned(),
                ));
            }
            let mut values = Vec::with_capacity(active);
            for index in 0..padded {
                let value = reader.i16()?;
                if index < active {
                    values.push(value);
                } else if value != 0 {
                    return Err(R4Error::InvalidBoundedEnvelope(
                        "bounded padding is nonzero".to_owned(),
                    ));
                }
            }
            tokens.push(BoundedToken::new(kind, values)?);
        }
        if reader.remaining() != 0 {
            return Err(R4Error::InvalidBoundedEnvelope(format!(
                "bounded envelope has {} trailing bytes",
                reader.remaining()
            )));
        }
        let view = Self {
            schema_version,
            arm,
            global,
            players,
            market,
            supplied_tile,
            accounting,
            tokens,
        };
        view.validate()?;
        if view.canonical_bytes()? != bytes {
            return Err(R4Error::InvalidBoundedEnvelope(
                "bounded envelope is not canonically encoded".to_owned(),
            ));
        }
        Ok(view)
    }

    pub fn canonical_blake3(&self) -> Result<String> {
        Ok(blake3::hash(&self.canonical_bytes()?).to_hex().to_string())
    }
}

#[derive(Default)]
struct TokenValues {
    values: Vec<i16>,
}

impl TokenValues {
    fn usize(&mut self, value: usize, field: &str) -> Result<()> {
        self.u32(
            u32::try_from(value)
                .map_err(|_| R4Error::InvalidBoundedView(format!("{field}={value} exceeds u32")))?,
            field,
        )
    }

    fn u8(&mut self, value: u8) {
        self.values.push(i16::from(value));
    }

    fn i8(&mut self, value: i8) {
        self.values.push(i16::from(value));
    }

    fn bool(&mut self, value: bool) {
        self.values.push(i16::from(value));
    }

    fn u16(&mut self, value: u16, field: &str) -> Result<()> {
        self.u32(u32::from(value), field)
    }

    fn u32(&mut self, value: u32, field: &str) -> Result<()> {
        self.values.push(i16::try_from(value).map_err(|_| {
            R4Error::InvalidBoundedView(format!("{field}={value} exceeds i16 feature range"))
        })?);
        Ok(())
    }

    fn bits16(&mut self, value: u16) {
        self.values.push(i16::from_le_bytes(value.to_le_bytes()));
    }

    fn u8s<const N: usize>(&mut self, values: &[u8; N]) {
        self.values.extend(values.iter().copied().map(i16::from));
    }

    fn u16s<const N: usize>(&mut self, values: &[u16; N], field: &str) -> Result<()> {
        for value in values {
            self.u16(*value, field)?;
        }
        Ok(())
    }

    fn u32s<const N: usize>(&mut self, values: &[u32; N], field: &str) -> Result<()> {
        for value in values {
            self.u32(*value, field)?;
        }
        Ok(())
    }

    fn finish(self, kind: BoundedTokenKind) -> Result<BoundedToken> {
        BoundedToken::new(kind, self.values)
    }
}

fn near_token(cell: &NearCell) -> Result<BoundedToken> {
    let mut values = TokenValues::default();
    values.u8(cell.index);
    values.i8(cell.relative_q);
    values.i8(cell.relative_r);
    match &cell.state {
        NearCellState::OutsideRules => values.u8(0),
        NearCellState::Empty => values.u8(1),
        NearCellState::Frontier(frontier) => {
            values.u8(2);
            values.u8(frontier.neighbor_presence_bits);
            values.u8s(&frontier.neighbor_facing_terrains);
            values.u8s(&frontier.adjacent_wildlife_counts);
            values.u8(frontier.occupied_neighbor_runs);
            values.u8(frontier.opposite_neighbor_pair_bits);
            values.usize(
                frontier.touched_habitat_components.len(),
                "near touched component count",
            )?;
            for touch in &frontier.touched_habitat_components {
                values.u8(touch.terrain);
                values.u16(touch.component_size, "near component size")?;
                values.u16(touch.near_member_count, "near component near count")?;
                values.u16(touch.far_member_count, "near component far count")?;
                values.u8(touch.contact_edge_bits);
            }
            values.u16s(&frontier.resulting_size_by_terrain, "near resulting size")?;
            values.u8(frontier.habitat_bridge_terrain_bits);
            values.u8(frontier.repeated_component_contact_terrain_bits);
            values.bool(frontier.supplied_tile_compatibility.is_some());
            if let Some(compatibility) = &frontier.supplied_tile_compatibility {
                let rotation_bits = compatibility.terrain_compatible_rotations.iter().try_fold(
                    0u8,
                    |bits, rotation| {
                        if *rotation >= 6 {
                            Err(R4Error::InvalidBoundedView(
                                "near compatibility rotation is invalid".to_owned(),
                            ))
                        } else {
                            Ok(bits | (1 << rotation))
                        }
                    },
                )?;
                values.u8(rotation_bits);
                values.u8(compatibility.best_matching_edge_count);
                values.usize(
                    compatibility.rotations.len(),
                    "near compatibility rotation count",
                )?;
                for rotation in &compatibility.rotations {
                    values.u8(rotation.rotation);
                    values.u8(rotation.matching_edge_bits);
                    values.u8(rotation.matching_edge_count);
                    values.bool(rotation.all_present_edges_match);
                    values.u16s(
                        &rotation.resulting_size_by_terrain,
                        "near rotation resulting size",
                    )?;
                }
            }
        }
        NearCellState::Occupied(occupied) => {
            values.u8(3);
            values.u8s(&occupied.semantic);
            values.u8s(&occupied.directed_edge_terrains);
        }
    }
    values.finish(BoundedTokenKind::NearCell)
}

fn habitat_component_token(component: &FarHabitatComponent) -> Result<BoundedToken> {
    let mut values = TokenValues::default();
    values.u8(component.relative_seat);
    values.u8(component.terrain);
    for (field, value) in [
        ("habitat member count", component.member_count),
        ("habitat near count", component.near_member_count),
        ("habitat far count", component.far_member_count),
        (
            "habitat internal edges",
            component.matching_internal_edge_count,
        ),
        ("habitat far edges", component.far_internal_edge_count),
        (
            "habitat crossing edges",
            component.near_far_crossing_edge_count,
        ),
        ("habitat open edges", component.open_boundary_edge_count),
        (
            "habitat frontier contacts",
            component.frontier_contact_count,
        ),
    ] {
        values.u16(value, field)?;
    }
    values.u16s(&component.degree_histogram, "habitat degree histogram")?;
    append_radial_summary(
        &mut values,
        component
            .radial_counts
            .iter()
            .map(|entry| (entry.distance, entry.count)),
    )?;
    append_sector_summary(
        &mut values,
        component
            .sector_counts
            .iter()
            .map(|entry| (entry.sector_bits, entry.count)),
    )?;
    append_bitmap(
        &mut values,
        component.local_member_indices.iter().copied(),
        "habitat local member",
    )?;
    append_bitmap(
        &mut values,
        component.portals.iter().map(|portal| portal.local_index),
        "habitat portal local index",
    )?;
    let mut portal_edges = [0u16; HEX_SECTORS];
    for portal in &component.portals {
        let value = portal_edges
            .get_mut(usize::from(portal.edge))
            .ok_or_else(|| R4Error::InvalidBoundedView("habitat portal edge invalid".to_owned()))?;
        checked_add_u16(value, 1, "habitat portal edge count")?;
    }
    values.u16s(&portal_edges, "habitat portal edge count")?;
    values.finish(BoundedTokenKind::HabitatComponent)
}

fn wildlife_component_token(component: &FarWildlifeComponent) -> Result<BoundedToken> {
    let mut values = TokenValues::default();
    values.u8(component.relative_seat);
    values.u8(component.wildlife);
    for (field, value) in [
        ("wildlife member count", component.member_count),
        ("wildlife near count", component.near_member_count),
        ("wildlife far count", component.far_member_count),
        ("wildlife internal edges", component.internal_edge_count),
        (
            "wildlife crossing edges",
            component.near_far_crossing_edge_count,
        ),
        ("wildlife endpoints", component.endpoint_count),
        ("wildlife branches", component.branch_count),
        ("wildlife diameter", component.graph_diameter),
    ] {
        values.u16(value, field)?;
    }
    values.u16s(&component.degree_histogram, "wildlife degree histogram")?;
    values.u16s(&component.edge_direction_counts, "wildlife direction edges")?;
    values.u16s(
        &component.max_collinear_run_by_axis,
        "wildlife collinear runs",
    )?;
    append_radial_summary(
        &mut values,
        component
            .radial_counts
            .iter()
            .map(|entry| (entry.distance, entry.count)),
    )?;
    append_sector_summary(
        &mut values,
        component
            .sector_counts
            .iter()
            .map(|entry| (entry.sector_bits, entry.count)),
    )?;
    append_bitmap(
        &mut values,
        component.local_member_indices.iter().copied(),
        "wildlife local member",
    )?;
    append_bitmap(
        &mut values,
        component.portals.iter().map(|portal| portal.local_index),
        "wildlife portal local index",
    )?;
    let mut portal_edges = [0u16; HEX_SECTORS];
    for portal in &component.portals {
        let value = portal_edges
            .get_mut(usize::from(portal.edge))
            .ok_or_else(|| {
                R4Error::InvalidBoundedView("wildlife portal edge invalid".to_owned())
            })?;
        checked_add_u16(value, 1, "wildlife portal edge count")?;
    }
    values.u16s(&portal_edges, "wildlife portal edge count")?;
    values.finish(BoundedTokenKind::WildlifeComponent)
}

#[derive(Default)]
struct WildlifeSummary {
    mass: u32,
    buckets: u32,
    radial: [u32; RADIAL_BINS],
    sectors: [u32; SECTOR_CLASSES],
    same_species: [u32; 7],
    occupied_neighbors: [u32; 7],
    adjacent_sums: [u32; WILDLIFE_TYPES],
    adjacent_nonzero: [u32; WILDLIFE_TYPES],
    diversity: [u32; 6],
    minimum_distance: Option<u16>,
    maximum_distance: u16,
}

impl WildlifeSummary {
    fn observe(&mut self, bucket: &FarWildlifeMotifBucket) -> Result<()> {
        let count = u32::from(bucket.count);
        checked_add_u32(&mut self.mass, count, "wildlife summary mass")?;
        checked_add_u32(&mut self.buckets, 1, "wildlife summary bucket count")?;
        let radial = self
            .radial
            .get_mut(radial_bin(bucket.signature.distance))
            .expect("radial_bin always returns an in-range index");
        checked_add_u32(radial, count, "wildlife radial histogram")?;
        let sector = self
            .sectors
            .get_mut(sector_class(bucket.signature.sector_bits)?)
            .expect("sector_class always returns an in-range index");
        checked_add_u32(sector, count, "wildlife sector histogram")?;
        let same_species = self
            .same_species
            .get_mut(usize::from(bucket.signature.same_species_neighbor_count))
            .ok_or_else(|| {
                R4Error::InvalidBoundedView(
                    "wildlife same-species neighbor count exceeds six".to_owned(),
                )
            })?;
        checked_add_u32(same_species, count, "wildlife neighbor histogram")?;
        let occupied_neighbors = self
            .occupied_neighbors
            .get_mut(usize::from(bucket.signature.occupied_neighbor_count))
            .ok_or_else(|| {
                R4Error::InvalidBoundedView(
                    "wildlife occupied-neighbor count exceeds six".to_owned(),
                )
            })?;
        checked_add_u32(
            occupied_neighbors,
            count,
            "wildlife occupied-neighbor histogram",
        )?;
        let mut diversity = 0usize;
        for (index, adjacent) in bucket.signature.adjacent_wildlife_counts.iter().enumerate() {
            let weighted = checked_mul_u32(
                count,
                u32::from(*adjacent),
                "wildlife adjacent weighted count",
            )?;
            checked_add_u32(
                &mut self.adjacent_sums[index],
                weighted,
                "wildlife adjacent sum",
            )?;
            if *adjacent > 0 {
                checked_add_u32(
                    &mut self.adjacent_nonzero[index],
                    count,
                    "wildlife adjacent nonzero count",
                )?;
                diversity += 1;
            }
        }
        let diversity_bin = self.diversity.get_mut(diversity).ok_or_else(|| {
            R4Error::InvalidBoundedView("wildlife diversity exceeds five".to_owned())
        })?;
        checked_add_u32(diversity_bin, count, "wildlife diversity histogram")?;
        self.minimum_distance = Some(
            self.minimum_distance
                .map_or(bucket.signature.distance, |value| {
                    value.min(bucket.signature.distance)
                }),
        );
        self.maximum_distance = self.maximum_distance.max(bucket.signature.distance);
        Ok(())
    }

    fn token(&self, relative_seat: u8, wildlife: u8) -> Result<BoundedToken> {
        let mut values = TokenValues::default();
        values.u8(relative_seat);
        values.u8(wildlife);
        values.u32(self.mass, "wildlife summary mass")?;
        values.u32(self.buckets, "wildlife summary bucket count")?;
        values.u16(
            self.minimum_distance.unwrap_or(0),
            "wildlife minimum distance",
        )?;
        values.u16(self.maximum_distance, "wildlife maximum distance")?;
        values.u32s(&self.radial, "wildlife radial histogram")?;
        values.u32s(&self.sectors, "wildlife sector histogram")?;
        values.u32s(&self.same_species, "wildlife neighbor histogram")?;
        values.u32s(
            &self.occupied_neighbors,
            "wildlife occupied-neighbor histogram",
        )?;
        values.u32s(&self.adjacent_sums, "wildlife adjacent sums")?;
        values.u32s(&self.adjacent_nonzero, "wildlife adjacent nonzero counts")?;
        values.u32s(&self.diversity, "wildlife diversity histogram")?;
        values.finish(BoundedTokenKind::WildlifeSummary)
    }
}

fn wildlife_summary_tokens(buckets: &[&FarWildlifeMotifBucket]) -> Result<Vec<BoundedToken>> {
    let mut summaries: [[WildlifeSummary; WILDLIFE_TYPES]; RELATIVE_SEATS] =
        std::array::from_fn(|_| std::array::from_fn(|_| WildlifeSummary::default()));
    for bucket in buckets {
        let seat = usize::from(bucket.signature.relative_seat);
        let wildlife = usize::from(bucket.signature.wildlife);
        let summary = summaries
            .get_mut(seat)
            .and_then(|row| row.get_mut(wildlife))
            .ok_or_else(|| {
                R4Error::InvalidBoundedView("wildlife bucket seat or species is invalid".to_owned())
            })?;
        summary.observe(bucket)?;
    }
    let mut tokens = Vec::with_capacity(RELATIVE_SEATS * WILDLIFE_TYPES);
    for (seat, row) in summaries.iter().enumerate() {
        for (wildlife, summary) in row.iter().enumerate() {
            tokens.push(summary.token(seat as u8, wildlife as u8)?);
        }
    }
    Ok(tokens)
}

#[derive(Default)]
struct FrontierSummary {
    source_mass: u32,
    source_buckets: u32,
    matched_mass: u32,
    matched_buckets: u32,
    radial: [u32; RADIAL_BINS],
    sectors: [u32; SECTOR_CLASSES],
    occupied_neighbors: [u32; 7],
    occupied_runs: [u32; 7],
    opposite_pairs: [u32; 4],
    facing_terrain: [u32; TERRAIN_TYPES],
    adjacent_wildlife: [u32; WILDLIFE_TYPES],
    touched_components: [u32; TERRAIN_TYPES],
    contact_edges: [u32; TERRAIN_TYPES],
    component_size_sum: [u32; TERRAIN_TYPES],
    component_size_max: [u32; TERRAIN_TYPES],
    component_near_sum: [u32; TERRAIN_TYPES],
    component_far_sum: [u32; TERRAIN_TYPES],
    resulting_size_sum: [u32; TERRAIN_TYPES],
    resulting_size_max: [u32; TERRAIN_TYPES],
    bridge_counts: [u32; TERRAIN_TYPES],
    repeated_counts: [u32; TERRAIN_TYPES],
    boundary_local_bitmap: [u16; NEAR_BITMAP_WORDS],
    boundary_edge_counts: [u32; HEX_SECTORS],
    minimum_distance: Option<u16>,
    maximum_distance: u16,
}

impl FrontierSummary {
    fn observe_source(&mut self, bucket: &FarFrontierBucket) -> Result<()> {
        checked_add_u32(
            &mut self.source_mass,
            u32::from(bucket.count),
            "frontier source mass",
        )?;
        checked_add_u32(&mut self.source_buckets, 1, "frontier source bucket count")
    }

    fn observe_matched(&mut self, bucket: &FarFrontierBucket) -> Result<()> {
        let count = u32::from(bucket.count);
        checked_add_u32(&mut self.matched_mass, count, "frontier matched mass")?;
        checked_add_u32(
            &mut self.matched_buckets,
            1,
            "frontier matched bucket count",
        )?;
        let radial = self
            .radial
            .get_mut(radial_bin(bucket.signature.distance))
            .expect("radial_bin always returns an in-range index");
        checked_add_u32(radial, count, "frontier radial histogram")?;
        let sector = self
            .sectors
            .get_mut(sector_class(bucket.signature.sector_bits)?)
            .expect("sector_class always returns an in-range index");
        checked_add_u32(sector, count, "frontier sector histogram")?;
        let occupied_neighbors = self
            .occupied_neighbors
            .get_mut(usize::from(bucket.signature.occupied_neighbor_count))
            .ok_or_else(|| {
                R4Error::InvalidBoundedView(
                    "frontier occupied-neighbor count exceeds six".to_owned(),
                )
            })?;
        checked_add_u32(
            occupied_neighbors,
            count,
            "frontier occupied-neighbor histogram",
        )?;
        let occupied_runs = self
            .occupied_runs
            .get_mut(usize::from(bucket.signature.occupied_neighbor_runs))
            .ok_or_else(|| {
                R4Error::InvalidBoundedView(
                    "frontier occupied-neighbor run count exceeds six".to_owned(),
                )
            })?;
        checked_add_u32(occupied_runs, count, "frontier run histogram")?;
        let opposite_pairs = self
            .opposite_pairs
            .get_mut(usize::from(bucket.signature.opposite_neighbor_pair_count))
            .ok_or_else(|| {
                R4Error::InvalidBoundedView(
                    "frontier opposite-neighbor pair count exceeds three".to_owned(),
                )
            })?;
        checked_add_u32(opposite_pairs, count, "frontier opposite-pair histogram")?;
        for terrain in 0..TERRAIN_TYPES {
            let facing = checked_mul_u32(
                count,
                u32::from(bucket.signature.facing_terrain_counts[terrain]),
                "frontier facing-terrain weighted count",
            )?;
            checked_add_u32(
                &mut self.facing_terrain[terrain],
                facing,
                "frontier facing-terrain count",
            )?;
            let adjacent = checked_mul_u32(
                count,
                u32::from(bucket.signature.adjacent_wildlife_counts[terrain]),
                "frontier adjacent-wildlife weighted count",
            )?;
            checked_add_u32(
                &mut self.adjacent_wildlife[terrain],
                adjacent,
                "frontier adjacent-wildlife count",
            )?;
            let resulting = u32::from(bucket.signature.resulting_size_by_terrain[terrain]);
            let weighted_resulting =
                checked_mul_u32(count, resulting, "frontier resulting-size weighted count")?;
            checked_add_u32(
                &mut self.resulting_size_sum[terrain],
                weighted_resulting,
                "frontier resulting-size sum",
            )?;
            self.resulting_size_max[terrain] = self.resulting_size_max[terrain].max(resulting);
            if bucket.signature.habitat_bridge_terrain_bits & (1 << terrain) != 0 {
                checked_add_u32(
                    &mut self.bridge_counts[terrain],
                    count,
                    "frontier bridge count",
                )?;
            }
            if bucket.signature.repeated_component_contact_terrain_bits & (1 << terrain) != 0 {
                checked_add_u32(
                    &mut self.repeated_counts[terrain],
                    count,
                    "frontier repeated-contact count",
                )?;
            }
        }
        for touch in &bucket.signature.touched_components {
            let terrain = usize::from(touch.terrain);
            if terrain >= TERRAIN_TYPES {
                return Err(R4Error::InvalidBoundedView(
                    "frontier touch terrain is invalid".to_owned(),
                ));
            }
            checked_add_u32(
                &mut self.touched_components[terrain],
                count,
                "frontier touched-component count",
            )?;
            let contact_edges = checked_mul_u32(
                count,
                u32::from(touch.contact_edge_count),
                "frontier contact-edge weighted count",
            )?;
            checked_add_u32(
                &mut self.contact_edges[terrain],
                contact_edges,
                "frontier contact-edge count",
            )?;
            let component_size = checked_mul_u32(
                count,
                u32::from(touch.component_size),
                "frontier component-size weighted count",
            )?;
            checked_add_u32(
                &mut self.component_size_sum[terrain],
                component_size,
                "frontier component-size sum",
            )?;
            self.component_size_max[terrain] =
                self.component_size_max[terrain].max(u32::from(touch.component_size));
            let component_near = checked_mul_u32(
                count,
                u32::from(touch.near_member_count),
                "frontier near-component weighted count",
            )?;
            checked_add_u32(
                &mut self.component_near_sum[terrain],
                component_near,
                "frontier near-component sum",
            )?;
            let component_far = checked_mul_u32(
                count,
                u32::from(touch.far_member_count),
                "frontier far-component weighted count",
            )?;
            checked_add_u32(
                &mut self.component_far_sum[terrain],
                component_far,
                "frontier far-component sum",
            )?;
        }
        for contact in &bucket.signature.boundary_contacts {
            set_bitmap(&mut self.boundary_local_bitmap, contact.local_index)?;
            let edge = self
                .boundary_edge_counts
                .get_mut(usize::from(contact.edge))
                .ok_or_else(|| {
                    R4Error::InvalidBoundedView(
                        "frontier boundary-contact edge is invalid".to_owned(),
                    )
                })?;
            checked_add_u32(edge, count, "frontier boundary-edge count")?;
        }
        self.minimum_distance = Some(
            self.minimum_distance
                .map_or(bucket.signature.distance, |value| {
                    value.min(bucket.signature.distance)
                }),
        );
        self.maximum_distance = self.maximum_distance.max(bucket.signature.distance);
        Ok(())
    }

    fn token(&self, relative_seat: u8, key: u8) -> Result<BoundedToken> {
        let mut values = TokenValues::default();
        values.u8(relative_seat);
        values.u8(key);
        values.u32(self.source_mass, "frontier source mass")?;
        values.u32(self.source_buckets, "frontier source buckets")?;
        values.u32(self.matched_mass, "frontier matched mass")?;
        values.u32(self.matched_buckets, "frontier matched buckets")?;
        values.u16(
            self.minimum_distance.unwrap_or(0),
            "frontier minimum distance",
        )?;
        values.u16(self.maximum_distance, "frontier maximum distance")?;
        values.u32s(&self.radial, "frontier radial histogram")?;
        values.u32s(&self.sectors, "frontier sector histogram")?;
        values.u32s(
            &self.occupied_neighbors,
            "frontier occupied-neighbor histogram",
        )?;
        values.u32s(&self.occupied_runs, "frontier run histogram")?;
        values.u32s(&self.opposite_pairs, "frontier opposite histogram")?;
        values.u32s(&self.facing_terrain, "frontier terrain counts")?;
        values.u32s(&self.adjacent_wildlife, "frontier wildlife counts")?;
        values.u32s(&self.touched_components, "frontier component counts")?;
        values.u32s(&self.contact_edges, "frontier contact edges")?;
        values.u32s(&self.component_size_sum, "frontier component size sum")?;
        values.u32s(&self.component_size_max, "frontier component size max")?;
        values.u32s(&self.component_near_sum, "frontier near component sum")?;
        values.u32s(&self.component_far_sum, "frontier far component sum")?;
        values.u32s(&self.resulting_size_sum, "frontier resulting size sum")?;
        values.u32s(&self.resulting_size_max, "frontier resulting size max")?;
        values.u32s(&self.bridge_counts, "frontier bridge counts")?;
        values.u32s(&self.repeated_counts, "frontier repeated counts")?;
        for word in self.boundary_local_bitmap {
            values.bits16(word);
        }
        values.u32s(&self.boundary_edge_counts, "frontier boundary edges")?;
        values.finish(BoundedTokenKind::FrontierSummary)
    }
}

fn frontier_summary_tokens(
    arm: BoundedArm,
    buckets: &[&FarFrontierBucket],
) -> Result<Vec<BoundedToken>> {
    let group_count = match arm {
        BoundedArm::SeatMarginal | BoundedArm::SelectiveExact => 1,
        BoundedArm::Directional => HEX_SECTORS,
        BoundedArm::Affordance => TERRAIN_TYPES,
    };
    let mut summaries = (0..RELATIVE_SEATS)
        .map(|_| {
            (0..group_count)
                .map(|_| FrontierSummary::default())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    for bucket in buckets {
        let seat = usize::from(bucket.signature.relative_seat);
        let row = summaries.get_mut(seat).ok_or_else(|| {
            R4Error::InvalidBoundedView("frontier bucket seat is invalid".to_owned())
        })?;
        for summary in row.iter_mut() {
            summary.observe_source(bucket)?;
        }
        match arm {
            BoundedArm::SeatMarginal | BoundedArm::SelectiveExact => {
                row[0].observe_matched(bucket)?;
            }
            BoundedArm::Directional => {
                for (sector, summary) in row.iter_mut().enumerate() {
                    if bucket.signature.sector_bits & (1 << sector) != 0 {
                        summary.observe_matched(bucket)?;
                    }
                }
            }
            BoundedArm::Affordance => {
                for (terrain, summary) in row.iter_mut().enumerate() {
                    if frontier_relevant_to_terrain(&bucket.signature, terrain) {
                        summary.observe_matched(bucket)?;
                    }
                }
            }
        }
    }
    let mut tokens = Vec::with_capacity(RELATIVE_SEATS * group_count);
    for (seat, row) in summaries.iter().enumerate() {
        for (key, summary) in row.iter().enumerate() {
            tokens.push(summary.token(seat as u8, key as u8)?);
        }
    }
    Ok(tokens)
}

fn exact_wildlife_bucket_token(bucket: &FarWildlifeMotifBucket) -> Result<BoundedToken> {
    let mut values = TokenValues::default();
    values.u8(bucket.signature.relative_seat);
    values.u8(bucket.signature.wildlife);
    values.u16(bucket.signature.distance, "exact wildlife distance")?;
    values.u8(bucket.signature.sector_bits);
    values.u8s(&bucket.signature.adjacent_wildlife_counts);
    values.u8(bucket.signature.same_species_neighbor_count);
    values.u8(bucket.signature.occupied_neighbor_count);
    values.u16(bucket.count, "exact wildlife count")?;
    values.finish(BoundedTokenKind::ExactWildlifeBucket)
}

fn exact_frontier_bucket_token(bucket: &FarFrontierBucket) -> Result<BoundedToken> {
    let mut values = TokenValues::default();
    values.u8(bucket.signature.relative_seat);
    values.u16(bucket.signature.distance, "exact frontier distance")?;
    values.u8(bucket.signature.sector_bits);
    values.u8(bucket.signature.occupied_neighbor_count);
    values.u8(bucket.signature.occupied_neighbor_runs);
    values.u8(bucket.signature.opposite_neighbor_pair_count);
    values.u8s(&bucket.signature.facing_terrain_counts);
    values.u8s(&bucket.signature.adjacent_wildlife_counts);
    values.usize(
        bucket.signature.touched_components.len(),
        "exact frontier touch count",
    )?;
    for touch in &bucket.signature.touched_components {
        values.u8(touch.terrain);
        values.u16(touch.component_size, "exact frontier component size")?;
        values.u16(touch.near_member_count, "exact frontier near count")?;
        values.u16(touch.far_member_count, "exact frontier far count")?;
        values.u8(touch.contact_edge_count);
    }
    values.u16s(
        &bucket.signature.resulting_size_by_terrain,
        "exact frontier resulting size",
    )?;
    values.u8(bucket.signature.habitat_bridge_terrain_bits);
    values.u8(bucket.signature.repeated_component_contact_terrain_bits);
    values.usize(
        bucket.signature.boundary_contacts.len(),
        "exact frontier boundary count",
    )?;
    for contact in &bucket.signature.boundary_contacts {
        values.u8(contact.local_index);
        values.u8(contact.edge);
    }
    values.u16(bucket.count, "exact frontier count")?;
    values.finish(BoundedTokenKind::ExactFrontierBucket)
}

fn select_wildlife_indices(
    buckets: &[FarWildlifeMotifBucket],
    limit: usize,
) -> Result<BTreeSet<usize>> {
    let mut grouped = BTreeMap::<_, Vec<usize>>::new();
    for (index, bucket) in buckets.iter().enumerate() {
        grouped
            .entry(wildlife_invariant_key(bucket))
            .or_default()
            .push(index);
    }
    let mut groups = grouped.into_iter().collect::<Vec<_>>();
    groups.sort_unstable_by(|(left_key, left), (right_key, right)| {
        wildlife_priority(&buckets[right[0]])
            .cmp(&wildlife_priority(&buckets[left[0]]))
            .then_with(|| left_key.cmp(right_key))
    });
    Ok(select_whole_groups(groups, limit))
}

fn wildlife_priority(bucket: &FarWildlifeMotifBucket) -> (u16, u8, u8, u8, std::cmp::Reverse<u16>) {
    (
        bucket.count,
        bucket.signature.same_species_neighbor_count,
        bucket.signature.occupied_neighbor_count,
        bucket
            .signature
            .adjacent_wildlife_counts
            .iter()
            .filter(|value| **value > 0)
            .count() as u8,
        std::cmp::Reverse(bucket.signature.distance),
    )
}

fn wildlife_invariant_key(bucket: &FarWildlifeMotifBucket) -> (u16, u8, u8, u16, [u8; 5], u8, u8) {
    (
        bucket.count,
        bucket.signature.relative_seat,
        bucket.signature.wildlife,
        bucket.signature.distance,
        bucket.signature.adjacent_wildlife_counts,
        bucket.signature.same_species_neighbor_count,
        bucket.signature.occupied_neighbor_count,
    )
}

fn select_frontier_indices(buckets: &[FarFrontierBucket], limit: usize) -> Result<BTreeSet<usize>> {
    let mut grouped = BTreeMap::<_, Vec<usize>>::new();
    for (index, bucket) in buckets.iter().enumerate() {
        grouped
            .entry(frontier_invariant_key(bucket))
            .or_default()
            .push(index);
    }
    let mut groups = grouped.into_iter().collect::<Vec<_>>();
    groups.sort_unstable_by(|(left_key, left), (right_key, right)| {
        frontier_priority(&buckets[right[0]])
            .cmp(&frontier_priority(&buckets[left[0]]))
            .then_with(|| left_key.cmp(right_key))
    });
    Ok(select_whole_groups(groups, limit))
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct FrontierInvariantKey {
    count: u16,
    relative_seat: u8,
    distance: u16,
    occupied_neighbor_count: u8,
    occupied_neighbor_runs: u8,
    opposite_neighbor_pair_count: u8,
    facing_terrain_counts: [u8; 5],
    adjacent_wildlife_counts: [u8; 5],
    touched_components: Vec<crate::model::FarFrontierTouch>,
    resulting_size_by_terrain: [u16; 5],
    habitat_bridge_terrain_bits: u8,
    repeated_component_contact_terrain_bits: u8,
    boundary_contact_count: usize,
}

fn frontier_priority(
    bucket: &FarFrontierBucket,
) -> (u16, u8, u8, u8, u16, u32, u8, std::cmp::Reverse<u16>) {
    (
        bucket.count,
        bucket.signature.boundary_contacts.len() as u8,
        bucket.signature.habitat_bridge_terrain_bits.count_ones() as u8,
        bucket
            .signature
            .repeated_component_contact_terrain_bits
            .count_ones() as u8,
        *bucket
            .signature
            .resulting_size_by_terrain
            .iter()
            .max()
            .unwrap_or(&0),
        bucket
            .signature
            .resulting_size_by_terrain
            .iter()
            .copied()
            .map(u32::from)
            .sum::<u32>(),
        bucket.signature.occupied_neighbor_count,
        std::cmp::Reverse(bucket.signature.distance),
    )
}

fn frontier_invariant_key(bucket: &FarFrontierBucket) -> FrontierInvariantKey {
    FrontierInvariantKey {
        count: bucket.count,
        relative_seat: bucket.signature.relative_seat,
        distance: bucket.signature.distance,
        occupied_neighbor_count: bucket.signature.occupied_neighbor_count,
        occupied_neighbor_runs: bucket.signature.occupied_neighbor_runs,
        opposite_neighbor_pair_count: bucket.signature.opposite_neighbor_pair_count,
        facing_terrain_counts: bucket.signature.facing_terrain_counts,
        adjacent_wildlife_counts: bucket.signature.adjacent_wildlife_counts,
        touched_components: bucket.signature.touched_components.clone(),
        resulting_size_by_terrain: bucket.signature.resulting_size_by_terrain,
        habitat_bridge_terrain_bits: bucket.signature.habitat_bridge_terrain_bits,
        repeated_component_contact_terrain_bits: bucket
            .signature
            .repeated_component_contact_terrain_bits,
        boundary_contact_count: bucket.signature.boundary_contacts.len(),
    }
}

fn select_whole_groups<K: Ord>(groups: Vec<(K, Vec<usize>)>, limit: usize) -> BTreeSet<usize> {
    let mut selected = BTreeSet::new();
    for (_, group) in groups {
        if selected.len().saturating_add(group.len()) <= limit {
            selected.extend(group);
        }
    }
    selected
}

fn frontier_relevant_to_terrain(signature: &FarFrontierSignature, terrain: usize) -> bool {
    signature.facing_terrain_counts[terrain] > 0
        || signature.resulting_size_by_terrain[terrain] > 0
        || signature.habitat_bridge_terrain_bits & (1 << terrain) != 0
        || signature.repeated_component_contact_terrain_bits & (1 << terrain) != 0
        || signature
            .touched_components
            .iter()
            .any(|touch| usize::from(touch.terrain) == terrain)
}

fn append_radial_summary(
    values: &mut TokenValues,
    entries: impl Iterator<Item = (u16, u16)>,
) -> Result<()> {
    let mut bins = [0u32; RADIAL_BINS];
    let mut minimum = None;
    let mut maximum = 0u16;
    let mut total = 0u32;
    let mut weighted_sum = 0u32;
    for (distance, count) in entries {
        let count = u32::from(count);
        checked_add_u32(&mut bins[radial_bin(distance)], count, "radial histogram")?;
        minimum = Some(minimum.map_or(distance, |value: u16| value.min(distance)));
        maximum = maximum.max(distance);
        checked_add_u32(&mut total, count, "radial total")?;
        let weighted = checked_mul_u32(count, u32::from(distance), "radial weighted count")?;
        checked_add_u32(&mut weighted_sum, weighted, "radial weighted sum")?;
    }
    values.u16(minimum.unwrap_or(0), "radial minimum")?;
    values.u16(maximum, "radial maximum")?;
    values.u32(total, "radial total")?;
    values.u32(weighted_sum, "radial weighted sum")?;
    values.u32s(&bins, "radial bins")?;
    Ok(())
}

fn append_sector_summary(
    values: &mut TokenValues,
    entries: impl Iterator<Item = (u8, u16)>,
) -> Result<()> {
    let mut sectors = [0u32; SECTOR_CLASSES];
    for (bits, count) in entries {
        checked_add_u32(
            &mut sectors[sector_class(bits)?],
            u32::from(count),
            "sector histogram",
        )?;
    }
    values.u32s(&sectors, "sector classes")
}

fn append_bitmap(
    values: &mut TokenValues,
    indices: impl Iterator<Item = u8>,
    field: &str,
) -> Result<()> {
    let mut bitmap = [0u16; NEAR_BITMAP_WORDS];
    for index in indices {
        set_bitmap(&mut bitmap, index).map_err(|_| {
            R4Error::InvalidBoundedView(format!("{field} index {index} is outside radius four"))
        })?;
    }
    for word in bitmap {
        values.bits16(word);
    }
    Ok(())
}

fn set_bitmap(bitmap: &mut [u16; NEAR_BITMAP_WORDS], index: u8) -> Result<()> {
    if usize::from(index) >= NearFieldRadius::Radius4.capacity() {
        return Err(R4Error::InvalidBoundedView(format!(
            "local index {index} is outside radius four"
        )));
    }
    let word = usize::from(index) / 16;
    let bit = usize::from(index) % 16;
    bitmap[word] |= 1 << bit;
    Ok(())
}

fn radial_bin(distance: u16) -> usize {
    usize::from(distance).min(RADIAL_BINS - 1)
}

fn sector_class(bits: u8) -> Result<usize> {
    if bits == 0 {
        return Ok(0);
    }
    for sector in 0..HEX_SECTORS {
        if bits == 1 << sector {
            return Ok(1 + sector);
        }
        let adjacent = (1 << sector) | (1 << ((sector + 1) % HEX_SECTORS));
        if bits == adjacent {
            return Ok(1 + HEX_SECTORS + sector);
        }
    }
    Err(R4Error::InvalidBoundedView(format!(
        "sector bits 0x{bits:02x} are outside the 12 D6 classes"
    )))
}

fn market_view(token: &MarketToken) -> BoundedMarketView {
    BoundedMarketView {
        slot: token.slot,
        tile: token.tile.map(tile_view),
        wildlife: token.wildlife.map(|wildlife| wildlife as u8),
    }
}

fn tile_view(tile: SuppliedTile) -> [u8; 4] {
    [
        tile.terrain_a as u8,
        tile.terrain_b.map_or(NONE, |terrain| terrain as u8),
        tile.wildlife_eligibility.bits(),
        u8::from(tile.keystone),
    ]
}

fn validate_market(market: &[BoundedMarketView]) -> Result<()> {
    if market.len() > 4 {
        return Err(R4Error::InvalidBoundedView(
            "bounded market contains more than four slots".to_owned(),
        ));
    }
    let mut seen = BTreeSet::new();
    for token in market {
        if token.slot >= 4 || !seen.insert(token.slot) {
            return Err(R4Error::InvalidBoundedView(
                "bounded market slots are invalid or duplicated".to_owned(),
            ));
        }
        if let Some(tile) = token.tile {
            validate_tile(tile, "market tile")?;
        }
        if token
            .wildlife
            .is_some_and(|wildlife| usize::from(wildlife) >= WILDLIFE_TYPES)
        {
            return Err(R4Error::InvalidBoundedView(
                "bounded market wildlife code is invalid".to_owned(),
            ));
        }
    }
    if market.windows(2).any(|pair| pair[0].slot >= pair[1].slot) {
        return Err(R4Error::InvalidBoundedView(
            "bounded market is not in canonical slot order".to_owned(),
        ));
    }
    Ok(())
}

fn validate_tile(tile: [u8; 4], field: &str) -> Result<()> {
    let [terrain_a, terrain_b, wildlife_bits, keystone] = tile;
    if usize::from(terrain_a) >= TERRAIN_TYPES
        || (terrain_b != NONE && usize::from(terrain_b) >= TERRAIN_TYPES)
        || wildlife_bits == 0
        || wildlife_bits & !0b1_1111 != 0
        || keystone > 1
        || (terrain_b == NONE && (keystone != 1 || wildlife_bits.count_ones() != 1))
        || (terrain_b != NONE && (terrain_a == terrain_b || keystone != 0))
    {
        return Err(R4Error::InvalidBoundedView(format!(
            "{field} has invalid Cascadia tile semantics"
        )));
    }
    Ok(())
}

fn validate_token_shape(token: &BoundedToken) -> Result<()> {
    let expected = match token.kind {
        BoundedTokenKind::NearCell => {
            if token.values.len() < 4 {
                return Err(R4Error::InvalidBoundedView(
                    "near-cell token is truncated".to_owned(),
                ));
            }
            return Ok(());
        }
        BoundedTokenKind::HabitatComponent => 64,
        BoundedTokenKind::WildlifeComponent => 70,
        BoundedTokenKind::WildlifeSummary => 65,
        BoundedTokenKind::FrontierSummary => 125,
        BoundedTokenKind::ExactWildlifeBucket => 12,
        BoundedTokenKind::ExactFrontierBucket => {
            if token.values.len() < 26 {
                return Err(R4Error::InvalidBoundedView(
                    "exact frontier token is truncated".to_owned(),
                ));
            }
            let touch_count = feature_usize(token, 16, "exact frontier touch count")?;
            let boundary_index = 17usize
                .checked_add(touch_count.checked_mul(5).ok_or_else(|| {
                    R4Error::InvalidBoundedView("exact frontier touch width overflow".to_owned())
                })?)
                .and_then(|value| value.checked_add(7))
                .ok_or_else(|| {
                    R4Error::InvalidBoundedView(
                        "exact frontier boundary offset overflow".to_owned(),
                    )
                })?;
            let boundary_count =
                feature_usize(token, boundary_index, "exact frontier boundary count")?;
            boundary_index
                .checked_add(2)
                .and_then(|value| {
                    boundary_count
                        .checked_mul(2)
                        .and_then(|width| value.checked_add(width))
                })
                .ok_or_else(|| {
                    R4Error::InvalidBoundedView("exact frontier token width overflow".to_owned())
                })?
        }
    };
    if token.values.len() != expected {
        return Err(R4Error::InvalidBoundedView(format!(
            "bounded token kind {} has active width {}; expected {expected}",
            token.kind.code(),
            token.values.len()
        )));
    }
    Ok(())
}

fn validate_summary_grids(view: &BoundedFeatureView) -> Result<()> {
    let near_tokens = view
        .tokens
        .iter()
        .filter(|token| token.kind == BoundedTokenKind::NearCell)
        .collect::<Vec<_>>();
    for (index, token) in near_tokens.iter().enumerate() {
        if feature_usize(token, 0, "near-cell index")? != index {
            return Err(R4Error::InvalidBoundedView(
                "near-cell token order or index is noncanonical".to_owned(),
            ));
        }
        if feature_usize(token, 3, "near-cell state code")? > 3 {
            return Err(R4Error::InvalidBoundedView(
                "near-cell state code is invalid".to_owned(),
            ));
        }
    }

    let wildlife_summaries = view
        .tokens
        .iter()
        .filter(|token| token.kind == BoundedTokenKind::WildlifeSummary)
        .collect::<Vec<_>>();
    let mut wildlife_mass = 0u32;
    let mut wildlife_buckets = 0u32;
    for (index, token) in wildlife_summaries.iter().enumerate() {
        let expected_seat = index / WILDLIFE_TYPES;
        let expected_wildlife = index % WILDLIFE_TYPES;
        if feature_usize(token, 0, "wildlife summary seat")? != expected_seat
            || feature_usize(token, 1, "wildlife summary species")? != expected_wildlife
        {
            return Err(R4Error::InvalidBoundedView(
                "wildlife summary grid is not canonical".to_owned(),
            ));
        }
        checked_add_u32(
            &mut wildlife_mass,
            feature_u32(token, 2, "wildlife summary mass")?,
            "wildlife summary accounting mass",
        )?;
        checked_add_u32(
            &mut wildlife_buckets,
            feature_u32(token, 3, "wildlife summary buckets")?,
            "wildlife summary accounting buckets",
        )?;
    }
    if wildlife_mass != view.accounting.summarized_wildlife_mass
        || wildlife_buckets != view.accounting.summarized_wildlife_buckets
    {
        return Err(R4Error::InvalidBoundedView(
            "wildlife summary fields disagree with accounting".to_owned(),
        ));
    }

    let frontier_group_count = match view.arm {
        BoundedArm::SeatMarginal | BoundedArm::SelectiveExact => 1,
        BoundedArm::Directional => HEX_SECTORS,
        BoundedArm::Affordance => TERRAIN_TYPES,
    };
    let frontier_summaries = view
        .tokens
        .iter()
        .filter(|token| token.kind == BoundedTokenKind::FrontierSummary)
        .collect::<Vec<_>>();
    let mut frontier_mass = 0u32;
    let mut frontier_buckets = 0u32;
    for (index, token) in frontier_summaries.iter().enumerate() {
        let expected_seat = index / frontier_group_count;
        let expected_key = index % frontier_group_count;
        if feature_usize(token, 0, "frontier summary seat")? != expected_seat
            || feature_usize(token, 1, "frontier summary key")? != expected_key
        {
            return Err(R4Error::InvalidBoundedView(
                "frontier summary grid is not canonical".to_owned(),
            ));
        }
        if expected_key == 0 {
            checked_add_u32(
                &mut frontier_mass,
                feature_u32(token, 2, "frontier source mass")?,
                "frontier summary accounting mass",
            )?;
            checked_add_u32(
                &mut frontier_buckets,
                feature_u32(token, 3, "frontier source buckets")?,
                "frontier summary accounting buckets",
            )?;
        }
    }
    if frontier_mass != view.accounting.summarized_frontier_mass
        || frontier_buckets != view.accounting.summarized_frontier_buckets
    {
        return Err(R4Error::InvalidBoundedView(
            "frontier summary fields disagree with accounting".to_owned(),
        ));
    }

    let exact_wildlife = view
        .tokens
        .iter()
        .filter(|token| token.kind == BoundedTokenKind::ExactWildlifeBucket)
        .collect::<Vec<_>>();
    let exact_frontier = view
        .tokens
        .iter()
        .filter(|token| token.kind == BoundedTokenKind::ExactFrontierBucket)
        .collect::<Vec<_>>();
    let exact_wildlife_mass = checked_feature_sum(
        &exact_wildlife,
        |token| token.values.len() - 1,
        "exact wildlife mass",
    )?;
    let exact_frontier_mass = checked_feature_sum(
        &exact_frontier,
        |token| token.values.len() - 1,
        "exact frontier mass",
    )?;
    if checked_len_u32(exact_wildlife.len(), "exact wildlife token count")?
        != view.accounting.exact_wildlife_buckets
        || exact_wildlife_mass != view.accounting.exact_wildlife_mass
        || checked_len_u32(exact_frontier.len(), "exact frontier token count")?
            != view.accounting.exact_frontier_buckets
        || exact_frontier_mass != view.accounting.exact_frontier_mass
    {
        return Err(R4Error::InvalidBoundedView(
            "exact token fields disagree with accounting".to_owned(),
        ));
    }
    Ok(())
}

fn feature_u32(token: &BoundedToken, index: usize, field: &str) -> Result<u32> {
    let value = *token.values.get(index).ok_or_else(|| {
        R4Error::InvalidBoundedView(format!("{field} is missing from bounded token"))
    })?;
    u32::try_from(value)
        .map_err(|_| R4Error::InvalidBoundedView(format!("{field} is negative in bounded token")))
}

fn feature_usize(token: &BoundedToken, index: usize, field: &str) -> Result<usize> {
    usize::try_from(feature_u32(token, index, field)?)
        .map_err(|_| R4Error::InvalidBoundedView(format!("{field} does not fit usize")))
}

fn checked_feature_sum(
    tokens: &[&BoundedToken],
    index: impl Fn(&BoundedToken) -> usize,
    field: &str,
) -> Result<u32> {
    let mut total = 0u32;
    for token in tokens {
        checked_add_u32(&mut total, feature_u32(token, index(token), field)?, field)?;
    }
    Ok(total)
}

fn metadata_scalar_count(view: &BoundedFeatureView) -> usize {
    10 + view.players.len() * 15
        + view
            .market
            .iter()
            .map(|market| 3 + usize::from(market.tile.is_some()) * 4)
            .sum::<usize>()
        + 1
        + usize::from(view.supplied_tile.is_some()) * 4
        + 12
}

fn accounting_values(accounting: &BoundedAccounting) -> [u32; 12] {
    [
        accounting.source_wildlife_buckets,
        accounting.source_wildlife_mass,
        accounting.summarized_wildlife_buckets,
        accounting.summarized_wildlife_mass,
        accounting.exact_wildlife_buckets,
        accounting.exact_wildlife_mass,
        accounting.source_frontier_buckets,
        accounting.source_frontier_mass,
        accounting.summarized_frontier_buckets,
        accounting.summarized_frontier_mass,
        accounting.exact_frontier_buckets,
        accounting.exact_frontier_mass,
    ]
}

fn checked_len_u32(value: usize, field: &str) -> Result<u32> {
    u32::try_from(value)
        .map_err(|_| R4Error::InvalidBoundedView(format!("{field}={value} exceeds u32")))
}

fn checked_count_sum(counts: impl Iterator<Item = u16>, field: &str) -> Result<u32> {
    let mut total = 0u32;
    for count in counts {
        checked_add_u32(&mut total, u32::from(count), field)?;
    }
    Ok(total)
}

fn checked_pair_sum(left: u32, right: u32, field: &str) -> Result<u32> {
    left.checked_add(right)
        .ok_or_else(|| R4Error::InvalidBoundedView(format!("{field} overflow")))
}

fn checked_add_u16(slot: &mut u16, increment: u16, field: &str) -> Result<()> {
    *slot = slot
        .checked_add(increment)
        .ok_or_else(|| R4Error::InvalidBoundedView(format!("{field} overflow")))?;
    Ok(())
}

fn checked_add_u32(slot: &mut u32, increment: u32, field: &str) -> Result<()> {
    *slot = slot
        .checked_add(increment)
        .ok_or_else(|| R4Error::InvalidBoundedView(format!("{field} overflow")))?;
    Ok(())
}

fn checked_mul_u32(left: u32, right: u32, field: &str) -> Result<u32> {
    left.checked_mul(right)
        .ok_or_else(|| R4Error::InvalidBoundedView(format!("{field} overflow")))
}

fn checked_u8(value: usize, field: &str) -> Result<u8> {
    u8::try_from(value).map_err(|_| R4Error::InvalidBoundedView(format!("{field} does not fit u8")))
}

fn checked_u16(value: usize, field: &str) -> Result<u16> {
    u16::try_from(value)
        .map_err(|_| R4Error::InvalidBoundedView(format!("{field} does not fit u16")))
}

#[derive(Default)]
struct EnvelopeWriter {
    bytes: Vec<u8>,
}

impl EnvelopeWriter {
    fn raw(&mut self, bytes: &[u8]) {
        self.bytes.extend_from_slice(bytes);
    }

    fn u8(&mut self, value: u8) {
        self.bytes.push(value);
    }

    fn u16(&mut self, value: u16) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn u32(&mut self, value: u32) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn i16(&mut self, value: i16) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }
}

struct EnvelopeReader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> EnvelopeReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn raw(&mut self, length: usize) -> Result<&'a [u8]> {
        let end = self
            .offset
            .checked_add(length)
            .ok_or_else(|| R4Error::InvalidBoundedEnvelope("bounded cursor overflow".to_owned()))?;
        let value = self.bytes.get(self.offset..end).ok_or_else(|| {
            R4Error::InvalidBoundedEnvelope("bounded envelope ended early".to_owned())
        })?;
        self.offset = end;
        Ok(value)
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.raw(1)?[0])
    }

    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.raw(2)?.try_into().unwrap()))
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.raw(4)?.try_into().unwrap()))
    }

    fn i16(&mut self) -> Result<i16> {
        Ok(i16::from_le_bytes(self.raw(2)?.try_into().unwrap()))
    }

    fn array_u8<const N: usize>(&mut self) -> Result<[u8; N]> {
        Ok(self.raw(N)?.try_into().unwrap())
    }

    fn remaining(&self) -> usize {
        self.bytes.len() - self.offset
    }
}

fn read_bool(reader: &mut EnvelopeReader<'_>) -> Result<bool> {
    match reader.u8()? {
        0 => Ok(false),
        1 => Ok(true),
        value => Err(R4Error::InvalidBoundedEnvelope(format!(
            "bounded boolean has value {value}"
        ))),
    }
}
