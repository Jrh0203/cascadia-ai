use std::collections::{BTreeMap, BTreeSet};

use cascadia_data::{
    BOARD_ENTITY_SIZE, CanonicalTileArchetype, ExactSemanticSupply, MARKET_ENTITY_SIZE,
    MAX_BOARD_TILES, PositionRecord, TARGET_DIM, standard_semantic_archetype_catalog,
};
use cascadia_game::{
    Board, D6Transform, DraftChoice, GameState, HexCoord, Market, MarketPrelude, Rotation,
    ScoreBreakdown, ScoringCards, ScoringVariant, Terrain, Tile, TileId, TurnAction, Wildlife,
    WildlifeMask, score_board,
};
use r2_sparse_entity_census::{
    FrontierToken, HabitatComponentToken, SparsePublicState, WildlifeMotifToken,
};
use serde::{Deserialize, Serialize};

use crate::{R3Error, Result};

pub const R3_ACTION_EDIT_SCHEMA_VERSION: u16 = 1;
pub const R3_STATE_TRUNK_SCHEMA_VERSION: u16 = 1;
pub const LOCAL_PATCH_MAX_RADIUS: u8 = 3;
const NONE: u8 = u8::MAX;
const SUPPLY_MAGIC: &[u8; 8] = b"CSR3SU1\0";
const SUPPLY_SCHEMA_VERSION: u16 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct AxialCoord {
    pub q: i16,
    pub r: i16,
}

impl AxialCoord {
    pub const ORIGIN: Self = Self { q: 0, r: 0 };

    pub const fn new(q: i16, r: i16) -> Self {
        Self { q, r }
    }

    pub const fn from_hex(coord: HexCoord) -> Self {
        Self::new(coord.q as i16, coord.r as i16)
    }

    pub fn to_hex(self) -> Result<HexCoord> {
        let q = i8::try_from(self.q)
            .map_err(|_| R3Error::Invariant(format!("q={} is outside i8", self.q)))?;
        let r = i8::try_from(self.r)
            .map_err(|_| R3Error::Invariant(format!("r={} is outside i8", self.r)))?;
        Ok(HexCoord::new(q, r))
    }

    pub const fn relative_to(self, center: Self) -> Self {
        Self::new(self.q - center.q, self.r - center.r)
    }

    pub fn transformed_offset(self, transform: D6Transform) -> Result<Self> {
        Ok(Self::from_hex(transform.transform_coord(self.to_hex()?)?))
    }

    pub fn distance(self) -> u16 {
        let q = i32::from(self.q);
        let r = i32::from(self.r);
        q.unsigned_abs()
            .max(r.unsigned_abs())
            .max((q + r).unsigned_abs()) as u16
    }
}

impl From<r2_sparse_entity_census::AxialCoord> for AxialCoord {
    fn from(value: r2_sparse_entity_census::AxialCoord) -> Self {
        Self::new(value.q, value.r)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TileSemantic {
    pub terrain_a: Terrain,
    pub terrain_b: Option<Terrain>,
    pub wildlife_eligibility: WildlifeMask,
    pub keystone: bool,
    pub semantic_archetype_id: Option<u16>,
}

impl TileSemantic {
    pub fn from_tile(tile: Tile) -> Self {
        Self::new(tile.terrain_a, tile.terrain_b, tile.wildlife, tile.keystone)
    }

    pub fn new(
        terrain_a: Terrain,
        terrain_b: Option<Terrain>,
        wildlife_eligibility: WildlifeMask,
        keystone: bool,
    ) -> Self {
        let tile = Tile {
            id: TileId(0),
            terrain_a,
            terrain_b,
            wildlife: wildlife_eligibility,
            keystone,
        };
        let semantic_archetype_id = standard_semantic_archetype_catalog()
            .id_for_archetype(CanonicalTileArchetype::from_tile(tile))
            .map(|id| id.code());
        Self {
            terrain_a,
            terrain_b,
            wildlife_eligibility,
            keystone,
            semantic_archetype_id,
        }
    }

    pub fn as_tile(&self) -> Tile {
        Tile {
            id: TileId(0),
            terrain_a: self.terrain_a,
            terrain_b: self.terrain_b,
            wildlife: self.wildlife_eligibility,
            keystone: self.keystone,
        }
    }

    fn validate(&self) -> Result<()> {
        if self.wildlife_eligibility == WildlifeMask::EMPTY {
            return Err(R3Error::Invariant(
                "tile semantic has an empty wildlife mask".to_owned(),
            ));
        }
        match (self.terrain_b, self.keystone) {
            (None, true) | (Some(_), false) => {}
            _ => {
                return Err(R3Error::Invariant(
                    "tile semantic does not match keystone/dual rules".to_owned(),
                ));
            }
        }
        if self.terrain_b == Some(self.terrain_a) {
            return Err(R3Error::Invariant(
                "dual tile repeats its primary terrain".to_owned(),
            ));
        }
        let expected = Self::new(
            self.terrain_a,
            self.terrain_b,
            self.wildlife_eligibility,
            self.keystone,
        );
        if self.semantic_archetype_id != expected.semantic_archetype_id {
            return Err(R3Error::Invariant(
                "tile semantic archetype reference is inconsistent".to_owned(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MarketSlotToken {
    pub slot: u8,
    pub tile: Option<TileSemantic>,
    pub wildlife: Option<Wildlife>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MarketSnapshot {
    pub slots: [MarketSlotToken; 4],
}

impl MarketSnapshot {
    pub fn from_market(market: &Market) -> Self {
        Self {
            slots: std::array::from_fn(|slot| MarketSlotToken {
                slot: slot as u8,
                tile: market.tiles[slot].map(TileSemantic::from_tile),
                wildlife: market.wildlife[slot],
            }),
        }
    }

    pub fn diff(&self, after: &Self) -> Vec<MarketSlotEdit> {
        self.slots
            .iter()
            .zip(&after.slots)
            .filter(|(before, after)| before != after)
            .map(|(before, after)| MarketSlotEdit {
                slot: before.slot,
                before: before.clone(),
                after: after.clone(),
            })
            .collect()
    }

    fn validate(&self) -> Result<()> {
        for (slot, token) in self.slots.iter().enumerate() {
            if usize::from(token.slot) != slot {
                return Err(R3Error::Invariant(
                    "market snapshot is not in canonical slot order".to_owned(),
                ));
            }
            if let Some(tile) = &token.tile {
                tile.validate()?;
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MarketSlotEdit {
    pub slot: u8,
    pub before: MarketSlotToken,
    pub after: MarketSlotToken,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SupplySnapshot {
    pub schema_version: u16,
    pub catalog_blake3: [u8; 32],
    pub wildlife_bag: [u16; 5],
    pub unseen_tile_count: u16,
    pub drawable_tile_count: u16,
    pub archetype_counts: Vec<u16>,
}

impl SupplySnapshot {
    pub fn from_exact(supply: &ExactSemanticSupply) -> Self {
        Self {
            schema_version: SUPPLY_SCHEMA_VERSION,
            catalog_blake3: *supply.catalog_blake3().as_bytes(),
            wildlife_bag: supply.wildlife_bag_counts(),
            unseen_tile_count: supply.unseen_tile_count(),
            drawable_tile_count: supply.drawable_tile_count(),
            archetype_counts: supply.archetype_counts().to_vec(),
        }
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>> {
        self.validate()?;
        let mut bytes = Vec::with_capacity(64 + self.archetype_counts.len() * 2);
        bytes.extend_from_slice(SUPPLY_MAGIC);
        bytes.extend_from_slice(&self.schema_version.to_le_bytes());
        bytes.extend_from_slice(&self.catalog_blake3);
        for count in self.wildlife_bag {
            bytes.extend_from_slice(&count.to_le_bytes());
        }
        bytes.extend_from_slice(&self.unseen_tile_count.to_le_bytes());
        bytes.extend_from_slice(&self.drawable_tile_count.to_le_bytes());
        bytes.extend_from_slice(&(self.archetype_counts.len() as u16).to_le_bytes());
        for count in &self.archetype_counts {
            bytes.extend_from_slice(&count.to_le_bytes());
        }
        Ok(bytes)
    }

    pub fn canonical_hash(&self) -> Result<[u8; 32]> {
        Ok(*blake3::hash(&self.canonical_bytes()?).as_bytes())
    }

    pub fn validate(&self) -> Result<()> {
        if self.schema_version != SUPPLY_SCHEMA_VERSION {
            return Err(R3Error::Invariant(format!(
                "unsupported supply snapshot schema {}",
                self.schema_version
            )));
        }
        let catalog = standard_semantic_archetype_catalog();
        if self.catalog_blake3 != *catalog.canonical_blake3().as_bytes() {
            return Err(R3Error::Invariant(
                "supply snapshot catalog identity mismatch".to_owned(),
            ));
        }
        if self.archetype_counts.len() != catalog.len() {
            return Err(R3Error::Invariant(
                "supply snapshot archetype vector has the wrong length".to_owned(),
            ));
        }
        let unseen = self
            .archetype_counts
            .iter()
            .map(|count| u32::from(*count))
            .sum::<u32>();
        if unseen != u32::from(self.unseen_tile_count) {
            return Err(R3Error::Invariant(
                "supply snapshot unseen count does not equal archetype sum".to_owned(),
            ));
        }
        if self.drawable_tile_count > self.unseen_tile_count {
            return Err(R3Error::Invariant(
                "drawable tile count exceeds unseen count".to_owned(),
            ));
        }
        if self.wildlife_bag.iter().any(|count| *count > 20) {
            return Err(R3Error::Invariant(
                "wildlife supply count exceeds the official multiplicity".to_owned(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SupplyCountDelta {
    pub archetype_id: u16,
    pub delta: i16,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SupplyDelta {
    pub before_blake3: [u8; 32],
    pub after_blake3: [u8; 32],
    pub wildlife_delta: [i16; 5],
    pub unseen_tile_delta: i16,
    pub drawable_tile_delta: i16,
    pub archetype_deltas: Vec<SupplyCountDelta>,
}

impl SupplyDelta {
    pub fn between(before: &SupplySnapshot, after: &SupplySnapshot) -> Result<Self> {
        before.validate()?;
        after.validate()?;
        if before.catalog_blake3 != after.catalog_blake3
            || before.archetype_counts.len() != after.archetype_counts.len()
        {
            return Err(R3Error::Invariant(
                "supply snapshots do not share one catalog".to_owned(),
            ));
        }
        let archetype_deltas = before
            .archetype_counts
            .iter()
            .zip(&after.archetype_counts)
            .enumerate()
            .filter_map(|(index, (left, right))| {
                let delta = i32::from(*right) - i32::from(*left);
                (delta != 0).then_some(SupplyCountDelta {
                    archetype_id: index as u16,
                    delta: delta as i16,
                })
            })
            .collect();
        Ok(Self {
            before_blake3: before.canonical_hash()?,
            after_blake3: after.canonical_hash()?,
            wildlife_delta: std::array::from_fn(|index| {
                i16::try_from(after.wildlife_bag[index]).unwrap()
                    - i16::try_from(before.wildlife_bag[index]).unwrap()
            }),
            unseen_tile_delta: after.unseen_tile_count as i16 - before.unseen_tile_count as i16,
            drawable_tile_delta: after.drawable_tile_count as i16
                - before.drawable_tile_count as i16,
            archetype_deltas,
        })
    }

    pub fn apply(&self, before: &SupplySnapshot) -> Result<SupplySnapshot> {
        self.validate()?;
        if before.canonical_hash()? != self.before_blake3 {
            return Err(R3Error::Invariant(
                "supply delta precondition hash mismatch".to_owned(),
            ));
        }
        let mut after = before.clone();
        for (value, delta) in after.wildlife_bag.iter_mut().zip(self.wildlife_delta) {
            *value = checked_add_u16(*value, delta, "wildlife supply")?;
        }
        after.unseen_tile_count = checked_add_u16(
            after.unseen_tile_count,
            self.unseen_tile_delta,
            "unseen tile supply",
        )?;
        after.drawable_tile_count = checked_add_u16(
            after.drawable_tile_count,
            self.drawable_tile_delta,
            "drawable tile supply",
        )?;
        let mut previous = None;
        for delta in &self.archetype_deltas {
            if previous.is_some_and(|prior| prior >= delta.archetype_id) {
                return Err(R3Error::Invariant(
                    "supply archetype deltas are not strictly ordered".to_owned(),
                ));
            }
            previous = Some(delta.archetype_id);
            let value = after
                .archetype_counts
                .get_mut(usize::from(delta.archetype_id))
                .ok_or_else(|| {
                    R3Error::Invariant("supply delta references an unknown archetype".to_owned())
                })?;
            *value = checked_add_u16(*value, delta.delta, "archetype supply")?;
        }
        after.validate()?;
        if after.canonical_hash()? != self.after_blake3 {
            return Err(R3Error::Invariant(
                "supply delta result hash mismatch".to_owned(),
            ));
        }
        Ok(after)
    }

    fn validate(&self) -> Result<()> {
        if self.archetype_deltas.iter().any(|delta| delta.delta == 0)
            || self
                .archetype_deltas
                .windows(2)
                .any(|pair| pair[0].archetype_id >= pair[1].archetype_id)
        {
            return Err(R3Error::Invariant(
                "supply archetype deltas are zero, duplicated, or noncanonical".to_owned(),
            ));
        }
        if self.archetype_deltas.iter().any(|delta| {
            usize::from(delta.archetype_id) >= standard_semantic_archetype_catalog().len()
        }) {
            return Err(R3Error::Invariant(
                "supply delta references an unknown semantic archetype".to_owned(),
            ));
        }
        Ok(())
    }
}

fn checked_add_u16(value: u16, delta: i16, label: &str) -> Result<u16> {
    let result = i32::from(value) + i32::from(delta);
    u16::try_from(result)
        .map_err(|_| R3Error::Invariant(format!("{label} delta underflowed or overflowed")))
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PublicStateTrunk {
    pub schema_version: u16,
    pub sparse: SparsePublicState,
    pub supply: SupplySnapshot,
}

pub struct PreparedPublicStateTrunk<'a> {
    trunk: &'a PublicStateTrunk,
    public_record: PositionRecord,
    packed_bytes: Vec<u8>,
    canonical_hash: [u8; 32],
}

impl PublicStateTrunk {
    pub fn observe(game: &GameState, game_index: u64) -> Result<Self> {
        let record = PositionRecord::observe(game, game_index);
        let sparse = SparsePublicState::from_position_record(&record, None)?;
        let supply = SupplySnapshot::from_exact(&ExactSemanticSupply::from_game(game)?);
        let trunk = Self {
            schema_version: R3_STATE_TRUNK_SCHEMA_VERSION,
            sparse,
            supply,
        };
        trunk.validate()?;
        Ok(trunk)
    }

    pub fn validate(&self) -> Result<()> {
        if self.schema_version != R3_STATE_TRUNK_SCHEMA_VERSION {
            return Err(R3Error::Invariant(format!(
                "unsupported R3 state trunk schema {}",
                self.schema_version
            )));
        }
        self.supply.validate()?;
        let targets = [0; TARGET_DIM];
        let reconstructed = self.sparse.reconstruct_position_record(targets)?;
        if reconstructed.targets != targets || !self.sparse.global.targets_omitted {
            return Err(R3Error::Invariant(
                "state trunk retained terminal targets".to_owned(),
            ));
        }
        Ok(())
    }

    pub fn public_record(&self) -> Result<PositionRecord> {
        self.sparse
            .reconstruct_position_record([0; TARGET_DIM])
            .map_err(Into::into)
    }

    pub fn token_count(&self) -> usize {
        1 + self.sparse.players.len()
            + self.sparse.market.len()
            + self.sparse.total_spatial_tokens()
            + 1
            + self.supply.archetype_counts.len()
    }

    pub fn prepare_action_edits(&self) -> Result<PreparedPublicStateTrunk<'_>> {
        self.validate()?;
        let public_record = self.public_record()?;
        let packed_bytes = self.to_packed_bytes()?;
        let canonical_hash = *blake3::hash(&packed_bytes).as_bytes();
        Ok(PreparedPublicStateTrunk {
            trunk: self,
            public_record,
            packed_bytes,
            canonical_hash,
        })
    }
}

impl PreparedPublicStateTrunk<'_> {
    pub fn packed_bytes(&self) -> &[u8] {
        &self.packed_bytes
    }

    pub const fn canonical_hash(&self) -> [u8; 32] {
        self.canonical_hash
    }

    pub const fn trunk(&self) -> &PublicStateTrunk {
        self.trunk
    }

    fn public_record(&self) -> &PositionRecord {
        &self.public_record
    }

    pub fn observe_legal_actions(
        &self,
        game: &GameState,
        prelude: &MarketPrelude,
    ) -> Result<Vec<(TurnAction, ActionEdit)>> {
        ActionEdit::observe_legal_actions_prepared(game, self, prelude)
    }

    pub fn observe_draft_actions(
        &self,
        game: &GameState,
        prelude: &MarketPrelude,
        draft: DraftChoice,
    ) -> Result<Vec<(TurnAction, ActionEdit)>> {
        ActionEdit::observe_actions_prepared(game, self, prelude, Some(draft))
    }

    pub fn canonical_transform_id(&self, edit: &ActionEdit) -> Result<u8> {
        edit.canonical_transform_id_prepared(self)
    }

    pub fn apply(&self, edit: &ActionEdit) -> Result<AppliedPublicState> {
        edit.apply_prepared(self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlayerPublicSummary {
    pub occupied_count: u8,
    pub nature_tokens: u8,
    pub wildlife_counts: [u8; 5],
    pub largest_habitats: [u8; 5],
}

impl PlayerPublicSummary {
    fn from_record(record: &PositionRecord, relative_seat: usize) -> Self {
        Self {
            occupied_count: record.board_counts[relative_seat],
            nature_tokens: record.nature_tokens[relative_seat],
            wildlife_counts: record.wildlife_counts[relative_seat],
            largest_habitats: record.habitat_sizes[relative_seat],
        }
    }

    fn write_to_record(&self, record: &mut PositionRecord, relative_seat: usize) {
        record.board_counts[relative_seat] = self.occupied_count;
        record.nature_tokens[relative_seat] = self.nature_tokens;
        record.wildlife_counts[relative_seat] = self.wildlife_counts;
        record.habitat_sizes[relative_seat] = self.largest_habitats;
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObjectUpdate<T> {
    pub before: T,
    pub after: T,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BoardTileToken {
    pub coord: AxialCoord,
    pub tile: TileSemantic,
    pub rotation: u8,
    pub directed_edge_terrains: [Terrain; 6],
    pub placed_wildlife: Option<Wildlife>,
}

impl BoardTileToken {
    fn from_entity(entity: [u8; BOARD_ENTITY_SIZE]) -> Result<Self> {
        let terrain_a = terrain(entity[2])?;
        let terrain_b = optional_terrain(entity[3])?;
        let rotation = Rotation::new(entity[4])
            .ok_or_else(|| R3Error::Invariant("board tile rotation is invalid".to_owned()))?;
        let wildlife_eligibility = WildlifeMask::from_bits(entity[5]);
        if wildlife_eligibility.bits() != entity[5] {
            return Err(R3Error::Invariant(
                "board tile wildlife mask is invalid".to_owned(),
            ));
        }
        let placed_wildlife = optional_wildlife(entity[6])?;
        let keystone = match entity[7] {
            0 => false,
            1 => true,
            _ => {
                return Err(R3Error::Invariant(
                    "board tile keystone flag is invalid".to_owned(),
                ));
            }
        };
        let tile = TileSemantic::new(terrain_a, terrain_b, wildlife_eligibility, keystone);
        tile.validate()?;
        if terrain_b.is_none() && rotation != Rotation::ZERO {
            return Err(R3Error::Invariant(
                "single-terrain board tile has noncanonical rotation".to_owned(),
            ));
        }
        if placed_wildlife.is_some_and(|wildlife| !wildlife_eligibility.contains(wildlife)) {
            return Err(R3Error::Invariant(
                "placed wildlife is unsupported by its tile".to_owned(),
            ));
        }
        Ok(Self {
            coord: AxialCoord::new(i16::from(entity[0] as i8), i16::from(entity[1] as i8)),
            directed_edge_terrains: std::array::from_fn(|edge| {
                tile.as_tile().terrain_on_edge(rotation, edge)
            }),
            tile,
            rotation: rotation.get(),
            placed_wildlife,
        })
    }

    fn to_entity(&self) -> Result<[u8; BOARD_ENTITY_SIZE]> {
        self.tile.validate()?;
        let rotation = Rotation::new(self.rotation)
            .ok_or_else(|| R3Error::Invariant("board tile rotation is invalid".to_owned()))?;
        if self.tile.terrain_b.is_none() && rotation != Rotation::ZERO {
            return Err(R3Error::Invariant(
                "single-terrain board tile has noncanonical rotation".to_owned(),
            ));
        }
        let expected_edges =
            std::array::from_fn(|edge| self.tile.as_tile().terrain_on_edge(rotation, edge));
        if self.directed_edge_terrains != expected_edges {
            return Err(R3Error::Invariant(
                "board tile directed edges disagree with its rotation".to_owned(),
            ));
        }
        let coord = self.coord.to_hex()?;
        Ok([
            coord.q as u8,
            coord.r as u8,
            self.tile.terrain_a as u8,
            self.tile.terrain_b.map_or(NONE, |value| value as u8),
            rotation.get(),
            self.tile.wildlife_eligibility.bits(),
            self.placed_wildlife.map_or(NONE, |value| value as u8),
            u8::from(self.tile.keystone),
        ])
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct BoardObjectChanges {
    pub added: Vec<BoardTileToken>,
    pub removed: Vec<BoardTileToken>,
    pub updated: Vec<ObjectUpdate<BoardTileToken>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComponentObject {
    pub object_key: [u8; 32],
    pub relative_seat: u8,
    pub terrain: Terrain,
    pub members: Vec<AxialCoord>,
    pub member_count: u16,
    pub matching_internal_edge_count: u16,
    pub open_boundary_edge_count: u16,
    pub frontier_contact_count: u16,
}

impl ComponentObject {
    fn from_r2(token: &HabitatComponentToken) -> Self {
        let members = token
            .members
            .iter()
            .copied()
            .map(Into::into)
            .collect::<Vec<_>>();
        let object_key = component_key(token.relative_seat, token.terrain, &members);
        Self {
            object_key,
            relative_seat: token.relative_seat,
            terrain: token.terrain,
            members,
            member_count: token.member_count,
            matching_internal_edge_count: token.matching_internal_edge_count,
            open_boundary_edge_count: token.open_boundary_edge_count,
            frontier_contact_count: token.frontier_contact_count,
        }
    }
}

fn component_key(relative_seat: u8, terrain: Terrain, members: &[AxialCoord]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"cascadia-r3-component-object-v1");
    hasher.update(&[relative_seat, terrain as u8]);
    hasher.update(&(members.len() as u16).to_le_bytes());
    for member in members {
        hasher.update(&member.q.to_le_bytes());
        hasher.update(&member.r.to_le_bytes());
    }
    *hasher.finalize().as_bytes()
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct ComponentChanges {
    pub added: Vec<ComponentObject>,
    pub removed: Vec<ComponentObject>,
    pub updated: Vec<ObjectUpdate<ComponentObject>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct FrontierChanges {
    pub added: Vec<FrontierToken>,
    pub removed: Vec<FrontierToken>,
    pub updated: Vec<ObjectUpdate<FrontierToken>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WildlifeMotifObject {
    pub relative_seat: u8,
    pub coord: AxialCoord,
    pub wildlife: Wildlife,
    pub neighbor_wildlife: [Option<Wildlife>; 6],
    pub adjacent_wildlife_counts: [u8; 5],
    pub same_species_neighbor_bits: u8,
}

impl WildlifeMotifObject {
    fn from_r2(token: &WildlifeMotifToken) -> Self {
        Self {
            relative_seat: token.relative_seat,
            coord: token.coord.into(),
            wildlife: token.wildlife,
            neighbor_wildlife: token.neighbor_wildlife,
            adjacent_wildlife_counts: token.adjacent_wildlife_counts,
            same_species_neighbor_bits: token.same_species_neighbor_bits,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct MotifChanges {
    pub added: Vec<WildlifeMotifObject>,
    pub removed: Vec<WildlifeMotifObject>,
    pub updated: Vec<ObjectUpdate<WildlifeMotifObject>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GlobalObjectReferences {
    pub before_component_keys: Vec<[u8; 32]>,
    pub after_component_keys: Vec<[u8; 32]>,
    pub frontier_coords: Vec<AxialCoord>,
    pub motif_coords: Vec<AxialCoord>,
    pub market_slots: Vec<u8>,
    pub supply_archetype_ids: Vec<u16>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DraftFactor {
    Paired { slot: u8 },
    Independent { tile_slot: u8, wildlife_slot: u8 },
}

impl DraftFactor {
    fn from_action(action: &TurnAction) -> Self {
        match action.draft {
            DraftChoice::Paired { slot } => Self::Paired {
                slot: slot.index() as u8,
            },
            DraftChoice::Independent {
                tile_slot,
                wildlife_slot,
            } => Self::Independent {
                tile_slot: tile_slot.index() as u8,
                wildlife_slot: wildlife_slot.index() as u8,
            },
        }
    }

    fn slots(&self) -> (u8, u8) {
        match *self {
            Self::Paired { slot } => (slot, slot),
            Self::Independent {
                tile_slot,
                wildlife_slot,
            } => (tile_slot, wildlife_slot),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActionFactors {
    pub replace_three_of_a_kind: bool,
    pub wildlife_wipe_masks: Vec<u8>,
    pub draft: DraftFactor,
    pub tile_destination: AxialCoord,
    pub tile_rotation: u8,
    pub tile_directed_edges: [Terrain; 6],
    pub wildlife_destination: Option<AxialCoord>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SelectedMarketObjects {
    pub tile_slot: u8,
    pub wildlife_slot: u8,
    pub tile: TileSemantic,
    pub wildlife: Wildlife,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PreludeEdit {
    pub market_before: MarketSnapshot,
    pub market_after: MarketSnapshot,
    pub market_edits: Vec<MarketSlotEdit>,
    pub active_player_before: PlayerPublicSummary,
    pub active_player_after: PlayerPublicSummary,
    pub supply: SupplyDelta,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlacementEdit {
    pub board: BoardObjectChanges,
    pub market_before: MarketSnapshot,
    pub market_after: MarketSnapshot,
    pub market_edits: Vec<MarketSlotEdit>,
    pub active_player_before: PlayerPublicSummary,
    pub active_player_after: PlayerPublicSummary,
    pub supply: SupplyDelta,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TurnAdvance {
    pub completed_turns_before: u8,
    pub completed_turns_after: u8,
    pub current_relative_seat_before: u8,
    pub current_relative_seat_after: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImmediateScoreDelta {
    pub habitat: [i16; 5],
    pub wildlife: [i16; 5],
    pub nature_tokens: i16,
    pub base_total: i16,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LocalPatchCell {
    pub offset: AxialCoord,
    pub inside_rules_grid: bool,
    pub frontier: bool,
    pub occupied: Option<CanonicalBoardToken>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CanonicalBoardToken {
    pub offset: AxialCoord,
    pub tile: TileSemantic,
    pub rotation: u8,
    pub directed_edge_terrains: [Terrain; 6],
    pub placed_wildlife: Option<Wildlife>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CanonicalLocalPatch {
    pub radius: u8,
    pub cells: Vec<LocalPatchCell>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CoverageByRadius {
    pub radius: u8,
    pub changed_coordinate_count: u16,
    pub covered_coordinate_count: u16,
    pub complete: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CanonicalComponentObject {
    pub terrain: Terrain,
    pub members: Vec<AxialCoord>,
    pub member_count: u16,
    pub matching_internal_edge_count: u16,
    pub open_boundary_edge_count: u16,
    pub frontier_contact_count: u16,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CanonicalFrontierTouch {
    pub terrain: Terrain,
    pub component_key: [u8; 32],
    pub component_size: u16,
    pub contact_edge_bits: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CanonicalFrontierToken {
    pub offset: AxialCoord,
    pub neighbor_presence_bits: u8,
    pub neighbor_facing_terrains: [Option<Terrain>; 6],
    pub adjacent_wildlife_counts: [u8; 5],
    pub occupied_neighbor_runs: u8,
    pub opposite_neighbor_pair_bits: u8,
    pub touched_habitat_components: Vec<CanonicalFrontierTouch>,
    pub resulting_size_by_terrain: [u16; 5],
    pub habitat_bridge_terrain_bits: u8,
    pub repeated_component_contact_terrain_bits: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CanonicalMotifObject {
    pub offset: AxialCoord,
    pub wildlife: Wildlife,
    pub neighbor_wildlife: [Option<Wildlife>; 6],
    pub adjacent_wildlife_counts: [u8; 5],
    pub same_species_neighbor_bits: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct CanonicalGlobalEdit {
    pub board_added: Vec<CanonicalBoardToken>,
    pub board_removed: Vec<CanonicalBoardToken>,
    pub board_updated: Vec<ObjectUpdate<CanonicalBoardToken>>,
    pub frontier_added: Vec<CanonicalFrontierToken>,
    pub frontier_removed: Vec<CanonicalFrontierToken>,
    pub frontier_updated: Vec<ObjectUpdate<CanonicalFrontierToken>>,
    pub components_added: Vec<CanonicalComponentObject>,
    pub components_removed: Vec<CanonicalComponentObject>,
    pub components_updated: Vec<ObjectUpdate<CanonicalComponentObject>>,
    pub motifs_added: Vec<CanonicalMotifObject>,
    pub motifs_removed: Vec<CanonicalMotifObject>,
    pub motifs_updated: Vec<ObjectUpdate<CanonicalMotifObject>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CanonicalActionView {
    pub local_patch: CanonicalLocalPatch,
    pub selected_tile: CanonicalBoardToken,
    pub wildlife_destination_offset: Option<AxialCoord>,
    pub global_edit: CanonicalGlobalEdit,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActionEdit {
    pub schema_version: u16,
    pub state_trunk_blake3: [u8; 32],
    pub expected_public_afterstate_blake3: [u8; 32],
    pub factors: ActionFactors,
    pub selected: SelectedMarketObjects,
    pub prelude: PreludeEdit,
    pub placement: PlacementEdit,
    pub turn: TurnAdvance,
    pub score_delta: ImmediateScoreDelta,
    pub frontier: FrontierChanges,
    pub components: ComponentChanges,
    pub motifs: MotifChanges,
    pub global_references: GlobalObjectReferences,
    pub canonical: CanonicalActionView,
    pub radius_coverage: [CoverageByRadius; 3],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppliedPublicState {
    pub record: PositionRecord,
    pub supply: SupplySnapshot,
}

struct CandidateContext {
    board: BoardObjectChanges,
    active_player_after: PlayerPublicSummary,
    frontier: FrontierChanges,
    components: ComponentChanges,
    motifs: MotifChanges,
    after_components: BTreeMap<u16, ComponentObject>,
    current_relative_seat_after: u8,
    score_delta: ImmediateScoreDelta,
}

impl ActionEdit {
    pub fn observe(
        game: &GameState,
        trunk: &PublicStateTrunk,
        action: &TurnAction,
    ) -> Result<Self> {
        let prepared = trunk.prepare_action_edits()?;
        let observed =
            Self::observe_actions_prepared(game, &prepared, &action.prelude(), Some(action.draft))?;
        let edit = observed
            .into_iter()
            .find_map(|(candidate, edit)| (candidate == *action).then_some(edit))
            .ok_or_else(|| {
                R3Error::Invariant(
                    "requested action was not emitted by authoritative legal enumeration"
                        .to_owned(),
                )
            })?;
        prepared.apply(&edit)?;
        Ok(edit)
    }

    pub fn observe_legal_actions(
        game: &GameState,
        trunk: &PublicStateTrunk,
        prelude: &MarketPrelude,
    ) -> Result<Vec<(TurnAction, Self)>> {
        trunk
            .prepare_action_edits()?
            .observe_legal_actions(game, prelude)
    }

    fn observe_legal_actions_prepared(
        game: &GameState,
        prepared: &PreparedPublicStateTrunk<'_>,
        prelude: &MarketPrelude,
    ) -> Result<Vec<(TurnAction, Self)>> {
        Self::observe_actions_prepared(game, prepared, prelude, None)
    }

    fn observe_actions_prepared(
        game: &GameState,
        prepared: &PreparedPublicStateTrunk<'_>,
        prelude: &MarketPrelude,
        only_draft: Option<DraftChoice>,
    ) -> Result<Vec<(TurnAction, Self)>> {
        let trunk = prepared.trunk();
        let game_index = trunk.sparse.global.game_index;
        let pre_record = PositionRecord::observe(game, game_index);
        let trunk_record = prepared.public_record();
        if normalized_record_bytes(&pre_record) != normalized_record_bytes(trunk_record) {
            return Err(R3Error::Invariant(
                "state trunk does not describe the supplied game".to_owned(),
            ));
        }

        let staged = game.preview_market_prelude(prelude)?;
        let staged_record = PositionRecord::observe(&staged, game_index);
        let pre_market = MarketSnapshot::from_market(game.market());
        let staged_market = MarketSnapshot::from_market(staged.market());
        let pre_supply = trunk.supply.clone();
        let staged_supply = SupplySnapshot::from_exact(&ExactSemanticSupply::from_game(&staged)?);
        let prelude_edit = PreludeEdit {
            market_before: pre_market.clone(),
            market_after: staged_market.clone(),
            market_edits: pre_market.diff(&staged_market),
            active_player_before: PlayerPublicSummary::from_record(&pre_record, 0),
            active_player_after: PlayerPublicSummary::from_record(&staged_record, 0),
            supply: SupplyDelta::between(&pre_supply, &staged_supply)?,
        };
        let before_score = score_board(
            &game.boards()[game.current_player()],
            game.config().scoring_cards,
        );
        let candidate_contexts = if let Some(draft) = only_draft {
            staged.evaluate_legal_draft_actions(&MarketPrelude::default(), draft, |board| {
                candidate_context(
                    &pre_record,
                    &staged_record,
                    &trunk.sparse,
                    board,
                    before_score,
                    game.config().scoring_cards,
                )
            })?
        } else {
            staged.evaluate_legal_turn_actions_with_context(
                &MarketPrelude::default(),
                |board, _, _, _| {
                    candidate_context(
                        &pre_record,
                        &staged_record,
                        &trunk.sparse,
                        board,
                        before_score,
                        game.config().scoring_cards,
                    )
                },
            )?
        };
        let state_trunk_blake3 = prepared.canonical_hash();
        let mut observed = Vec::with_capacity(candidate_contexts.len());
        for (mut action, candidate) in candidate_contexts {
            action.replace_three_of_a_kind = prelude.replace_three_of_a_kind;
            action.wildlife_wipes.clone_from(&prelude.wildlife_wipes);
            let candidate = candidate?;
            let factors = action_factors(&action, &staged_market)?;
            let selected = selected_market_objects(&factors.draft, &staged_market)?;
            let after_market = market_after_draft(&staged_market, &factors.draft)?;
            let after_supply =
                supply_after_draft(&staged_supply, selected.wildlife, action.wildlife.is_some())?;
            let placement = PlacementEdit {
                board: candidate.board,
                market_before: staged_market.clone(),
                market_after: after_market.clone(),
                market_edits: staged_market.diff(&after_market),
                active_player_before: PlayerPublicSummary::from_record(&staged_record, 0),
                active_player_after: candidate.active_player_after,
                supply: SupplyDelta::between(&staged_supply, &after_supply)?,
            };
            let mut after_record = staged_record.clone();
            apply_board_changes(&mut after_record, &placement.board)?;
            placement
                .active_player_after
                .write_to_record(&mut after_record, 0);
            write_market_snapshot(&mut after_record, &after_market)?;
            after_record.turn = after_record.turn.checked_add(1).ok_or_else(|| {
                R3Error::Invariant("completed-turn counter overflowed".to_owned())
            })?;
            after_record.targets = [0; TARGET_DIM];
            let turn = TurnAdvance {
                completed_turns_before: pre_record.turn,
                completed_turns_after: after_record.turn,
                current_relative_seat_before: trunk.sparse.global.current_relative_seat,
                current_relative_seat_after: candidate.current_relative_seat_after,
            };
            let global_references = global_references(
                &prelude_edit,
                &placement,
                &candidate.frontier,
                &candidate.components,
                &candidate.motifs,
            );
            let canonical = canonical_action_view(
                &trunk.sparse,
                &candidate.after_components,
                &factors,
                &selected,
                &placement.board,
                &candidate.frontier,
                &candidate.components,
                &candidate.motifs,
            )?;
            let radius_coverage = radius_coverage(&factors, &candidate.frontier, &candidate.motifs);
            let edit = Self {
                schema_version: R3_ACTION_EDIT_SCHEMA_VERSION,
                state_trunk_blake3,
                expected_public_afterstate_blake3: record_hash(&after_record),
                factors,
                selected,
                prelude: prelude_edit.clone(),
                placement,
                turn,
                score_delta: candidate.score_delta,
                frontier: candidate.frontier,
                components: candidate.components,
                motifs: candidate.motifs,
                global_references,
                canonical,
                radius_coverage,
            };
            edit.validate()?;
            observed.push((action, edit));
        }
        if !game.is_game_over() && observed.is_empty() {
            return Err(R3Error::Invariant(
                "authoritative legal enumeration emitted no actions".to_owned(),
            ));
        }
        Ok(observed)
    }

    pub fn apply(&self, trunk: &PublicStateTrunk) -> Result<AppliedPublicState> {
        let prepared = trunk.prepare_action_edits()?;
        self.apply_prepared(&prepared)
    }

    pub fn canonical_transform_id(&self, trunk: &PublicStateTrunk) -> Result<u8> {
        let prepared = trunk.prepare_action_edits()?;
        self.canonical_transform_id_prepared(&prepared)
    }

    fn canonical_transform_id_prepared(
        &self,
        prepared: &PreparedPublicStateTrunk<'_>,
    ) -> Result<u8> {
        let applied = self.apply_prepared(prepared)?;
        let mut geometry_record = applied.record;
        geometry_record.market_entities = prepared.public_record().market_entities;
        let after_sparse = SparsePublicState::from_position_record(&geometry_record, None)?;
        let after_components = component_lookup(&after_sparse);
        let tile = self.selected.tile.as_tile();
        let rotation = Rotation::new(self.factors.tile_rotation)
            .ok_or_else(|| R3Error::Invariant("invalid action rotation".to_owned()))?;
        let mut matching = Vec::new();
        for transform in D6Transform::ALL {
            if transform.transform_tile_rotation(tile, rotation) != Rotation::ZERO {
                continue;
            }
            let candidate = canonical_action_view_for_transform(
                &prepared.trunk().sparse,
                &after_components,
                &self.factors,
                &self.selected,
                &self.placement.board,
                &self.frontier,
                &self.components,
                &self.motifs,
                transform,
            )?;
            if candidate == self.canonical {
                matching.push(transform.id());
            }
        }
        matching.sort_unstable();
        matching.dedup();
        matching.first().copied().ok_or_else(|| {
            R3Error::Invariant(
                "stored canonical action view has no authoritative D6 transform".to_owned(),
            )
        })
    }

    fn apply_prepared(
        &self,
        prepared: &PreparedPublicStateTrunk<'_>,
    ) -> Result<AppliedPublicState> {
        let trunk = prepared.trunk();
        self.validate()?;
        if prepared.canonical_hash() != self.state_trunk_blake3 {
            return Err(R3Error::Invariant(
                "action edit state-trunk precondition mismatch".to_owned(),
            ));
        }
        let mut record = prepared.public_record().clone();
        let before_record = record.clone();
        if trunk.sparse.global.current_relative_seat != self.turn.current_relative_seat_before {
            return Err(R3Error::Invariant(
                "turn-advance current-seat precondition mismatch".to_owned(),
            ));
        }
        if market_snapshot_from_record(&record)? != self.prelude.market_before {
            return Err(R3Error::Invariant(
                "prelude market snapshot precondition mismatch".to_owned(),
            ));
        }
        apply_market_edits(&mut record, &self.prelude.market_edits)?;
        if market_snapshot_from_record(&record)? != self.prelude.market_after {
            return Err(R3Error::Invariant(
                "prelude market edits did not reproduce the staged market".to_owned(),
            ));
        }
        if PlayerPublicSummary::from_record(&record, 0) != self.prelude.active_player_before {
            return Err(R3Error::Invariant(
                "prelude active-player precondition mismatch".to_owned(),
            ));
        }
        self.prelude
            .active_player_after
            .write_to_record(&mut record, 0);
        let staged_supply = self.prelude.supply.apply(&trunk.supply)?;

        if PlayerPublicSummary::from_record(&record, 0) != self.placement.active_player_before {
            return Err(R3Error::Invariant(
                "placement active-player precondition mismatch".to_owned(),
            ));
        }
        if market_snapshot_from_record(&record)? != self.placement.market_before {
            return Err(R3Error::Invariant(
                "placement market snapshot precondition mismatch".to_owned(),
            ));
        }
        apply_board_changes(&mut record, &self.placement.board)?;
        apply_market_edits(&mut record, &self.placement.market_edits)?;
        if market_snapshot_from_record(&record)? != self.placement.market_after {
            return Err(R3Error::Invariant(
                "placement market edits did not reproduce the public afterstate".to_owned(),
            ));
        }
        self.placement
            .active_player_after
            .write_to_record(&mut record, 0);
        let supply = self.placement.supply.apply(&staged_supply)?;

        if record.turn != self.turn.completed_turns_before {
            return Err(R3Error::Invariant(
                "turn-advance precondition mismatch".to_owned(),
            ));
        }
        record.turn = self.turn.completed_turns_after;
        record.targets = [0; TARGET_DIM];
        if record_hash(&record) != self.expected_public_afterstate_blake3 {
            return Err(R3Error::Invariant(
                "action edit result does not match authoritative afterstate hash".to_owned(),
            ));
        }

        let after_sparse = sparse_geometry_afterstate(&record, &before_record)?;
        if after_sparse.global.current_relative_seat != self.turn.current_relative_seat_after {
            return Err(R3Error::Invariant(
                "turn-advance current-seat result mismatch".to_owned(),
            ));
        }
        let frontier = frontier_changes(&trunk.sparse, &after_sparse)?;
        let components = component_changes(&trunk.sparse, &after_sparse);
        let motifs = motif_changes(&trunk.sparse, &after_sparse);
        if frontier != self.frontier || components != self.components || motifs != self.motifs {
            return Err(R3Error::Invariant(
                "stored global edit differs from regenerated exact geometry".to_owned(),
            ));
        }
        if global_references(
            &self.prelude,
            &self.placement,
            &frontier,
            &components,
            &motifs,
        ) != self.global_references
        {
            return Err(R3Error::Invariant(
                "stored global references differ from regenerated references".to_owned(),
            ));
        }
        if radius_coverage(&self.factors, &frontier, &motifs) != self.radius_coverage {
            return Err(R3Error::Invariant(
                "stored radius coverage differs from regenerated coverage".to_owned(),
            ));
        }
        if score_delta_from_records(&before_record, &record)? != self.score_delta {
            return Err(R3Error::Invariant(
                "stored score delta differs from the edited public board".to_owned(),
            ));
        }
        let after_components = component_lookup(&after_sparse);
        let canonical = canonical_action_view(
            &trunk.sparse,
            &after_components,
            &self.factors,
            &self.selected,
            &self.placement.board,
            &frontier,
            &components,
            &motifs,
        )?;
        if canonical != self.canonical {
            return Err(R3Error::Invariant(
                "stored canonical action view differs from regenerated view".to_owned(),
            ));
        }
        Ok(AppliedPublicState { record, supply })
    }

    pub fn validate(&self) -> Result<()> {
        if self.schema_version != R3_ACTION_EDIT_SCHEMA_VERSION {
            return Err(R3Error::Invariant(format!(
                "unsupported action-edit schema {}",
                self.schema_version
            )));
        }
        self.selected.tile.validate()?;
        self.prelude.market_before.validate()?;
        self.prelude.market_after.validate()?;
        self.placement.market_before.validate()?;
        self.placement.market_after.validate()?;
        self.prelude.supply.validate()?;
        self.placement.supply.validate()?;
        if self.prelude.supply.after_blake3 != self.placement.supply.before_blake3 {
            return Err(R3Error::Invariant(
                "prelude and placement supply hashes do not form one edit chain".to_owned(),
            ));
        }
        validate_supply_delta_semantics(
            &self.prelude.supply,
            &self.prelude.market_before,
            &self.prelude.market_after,
            None,
        )?;
        validate_supply_delta_semantics(
            &self.placement.supply,
            &self.placement.market_before,
            &self.placement.market_after,
            Some(&self.placement.board),
        )?;
        if self.prelude.market_after != self.placement.market_before {
            return Err(R3Error::Invariant(
                "prelude and placement staged markets disagree".to_owned(),
            ));
        }
        if self.prelude.market_before.diff(&self.prelude.market_after) != self.prelude.market_edits
            || self
                .placement
                .market_before
                .diff(&self.placement.market_after)
                != self.placement.market_edits
        {
            return Err(R3Error::Invariant(
                "stored market edits are not the exact snapshot differences".to_owned(),
            ));
        }
        if self.prelude.active_player_after != self.placement.active_player_before {
            return Err(R3Error::Invariant(
                "prelude and placement active-player summaries disagree".to_owned(),
            ));
        }
        if self
            .factors
            .wildlife_wipe_masks
            .iter()
            .any(|mask| *mask == 0 || *mask & !0x0f != 0)
        {
            return Err(R3Error::Invariant(
                "wildlife wipe mask is empty or references a non-market slot".to_owned(),
            ));
        }
        if Rotation::new(self.factors.tile_rotation).is_none() {
            return Err(R3Error::Invariant(
                "action factor tile rotation is invalid".to_owned(),
            ));
        }
        let (tile_slot, wildlife_slot) = self.factors.draft.slots();
        if tile_slot >= 4
            || wildlife_slot >= 4
            || self.selected.tile_slot != tile_slot
            || self.selected.wildlife_slot != wildlife_slot
        {
            return Err(R3Error::Invariant(
                "selected market objects disagree with draft factors".to_owned(),
            ));
        }
        if selected_market_objects(&self.factors.draft, &self.placement.market_before)?
            != self.selected
            || market_after_draft(&self.placement.market_before, &self.factors.draft)?
                != self.placement.market_after
        {
            return Err(R3Error::Invariant(
                "selected objects or after-market disagree with the draft".to_owned(),
            ));
        }
        let rotation = Rotation::new(self.factors.tile_rotation).unwrap();
        if self.factors.tile_directed_edges
            != std::array::from_fn(|edge| {
                self.selected.tile.as_tile().terrain_on_edge(rotation, edge)
            })
        {
            return Err(R3Error::Invariant(
                "action directed edges disagree with selected tile".to_owned(),
            ));
        }
        validate_board_action(self)?;
        validate_player_changes(self)?;
        if self.turn.completed_turns_after
            != self
                .turn
                .completed_turns_before
                .checked_add(1)
                .ok_or_else(|| R3Error::Invariant("completed-turn counter overflowed".to_owned()))?
        {
            return Err(R3Error::Invariant(
                "turn advance must contain exactly one completed action".to_owned(),
            ));
        }
        if self.canonical.local_patch.radius != LOCAL_PATCH_MAX_RADIUS
            || self.canonical.local_patch.cells.len() != 37
        {
            return Err(R3Error::Invariant(
                "canonical radius-3 patch must contain exactly 37 cells".to_owned(),
            ));
        }
        let offsets = self
            .canonical
            .local_patch
            .cells
            .iter()
            .map(|cell| cell.offset)
            .collect::<BTreeSet<_>>();
        if offsets.len() != 37
            || offsets
                .iter()
                .any(|offset| offset.distance() > u16::from(LOCAL_PATCH_MAX_RADIUS))
        {
            return Err(R3Error::Invariant(
                "canonical local patch has duplicate or out-of-radius cells".to_owned(),
            ));
        }
        if self
            .canonical
            .local_patch
            .cells
            .windows(2)
            .any(|pair| pair[0].offset >= pair[1].offset)
            || self.canonical.local_patch.cells.iter().any(|cell| {
                cell.occupied
                    .as_ref()
                    .is_some_and(|tile| tile.offset != cell.offset)
            })
        {
            return Err(R3Error::Invariant(
                "canonical local patch is not strictly ordered or self-consistent".to_owned(),
            ));
        }
        if self.canonical.selected_tile.offset != AxialCoord::ORIGIN
            || self.canonical.selected_tile.rotation != Rotation::ZERO.get()
            || self.canonical.selected_tile.tile != self.selected.tile
            || self.canonical.selected_tile.directed_edge_terrains
                != std::array::from_fn(|edge| {
                    self.selected
                        .tile
                        .as_tile()
                        .terrain_on_edge(Rotation::ZERO, edge)
                })
        {
            return Err(R3Error::Invariant(
                "canonical selected tile is not centered in its zero-rotation frame".to_owned(),
            ));
        }
        if self.radius_coverage.map(|coverage| coverage.radius) != [1, 2, 3] {
            return Err(R3Error::Invariant(
                "radius coverage rows are not exactly 1, 2, and 3".to_owned(),
            ));
        }
        validate_sorted_unique_edits(&self.prelude.market_edits)?;
        validate_sorted_unique_edits(&self.placement.market_edits)?;
        validate_world_edit_order(self)?;
        validate_canonical_edit_order(&self.canonical.global_edit)?;
        validate_global_reference_order(&self.global_references)?;
        Ok(())
    }

    pub fn token_count(&self) -> usize {
        let geometry = &self.canonical.global_edit;
        1 + self.canonical.local_patch.cells.len()
            + self.factors.wildlife_wipe_masks.len()
            + self.prelude.market_edits.len()
            + self.placement.market_edits.len()
            + self.prelude.supply.archetype_deltas.len()
            + self.placement.supply.archetype_deltas.len()
            + geometry.board_added.len()
            + geometry.board_removed.len()
            + geometry.board_updated.len()
            + geometry.frontier_added.len()
            + geometry.frontier_removed.len()
            + geometry.frontier_updated.len()
            + geometry.components_added.len()
            + geometry.components_removed.len()
            + geometry.components_updated.len()
            + geometry.motifs_added.len()
            + geometry.motifs_removed.len()
            + geometry.motifs_updated.len()
    }
}

fn candidate_context(
    pre_record: &PositionRecord,
    staged_record: &PositionRecord,
    before_sparse: &SparsePublicState,
    board: &Board,
    before_score: ScoreBreakdown,
    scoring_cards: cascadia_game::ScoringCards,
) -> Result<CandidateContext> {
    let mut geometry_record = staged_record.clone();
    write_active_board(&mut geometry_record, board);
    geometry_record.turn = geometry_record
        .turn
        .checked_add(1)
        .ok_or_else(|| R3Error::Invariant("completed-turn counter overflowed".to_owned()))?;
    geometry_record.targets = [0; TARGET_DIM];
    let after_sparse = sparse_geometry_afterstate(&geometry_record, pre_record)?;
    let frontier = frontier_changes(before_sparse, &after_sparse)?;
    let components = component_changes(before_sparse, &after_sparse);
    let motifs = motif_changes(before_sparse, &after_sparse);
    Ok(CandidateContext {
        board: board_changes(pre_record, &geometry_record)?,
        active_player_after: PlayerPublicSummary::from_record(&geometry_record, 0),
        frontier,
        components,
        motifs,
        after_components: component_lookup(&after_sparse),
        current_relative_seat_after: after_sparse.global.current_relative_seat,
        score_delta: score_delta_between(before_score, score_board(board, scoring_cards)),
    })
}

fn action_factors(action: &TurnAction, staged_market: &MarketSnapshot) -> Result<ActionFactors> {
    let draft = DraftFactor::from_action(action);
    let selected = selected_market_objects(&draft, staged_market)?;
    let tile = selected.tile.as_tile();
    let requested_rotation = action.tile.rotation;
    let tile_rotation = tile.canonical_rotation(requested_rotation);
    if tile_rotation != requested_rotation {
        return Err(R3Error::Invariant(
            "action uses a noncanonical tile rotation".to_owned(),
        ));
    }
    Ok(ActionFactors {
        replace_three_of_a_kind: action.replace_three_of_a_kind,
        wildlife_wipe_masks: action
            .wildlife_wipes
            .iter()
            .map(|wipe| {
                wipe.slots
                    .iter()
                    .fold(0u8, |mask, slot| mask | (1 << slot.index()))
            })
            .collect(),
        draft,
        tile_destination: AxialCoord::from_hex(action.tile.coord),
        tile_rotation: tile_rotation.get(),
        tile_directed_edges: std::array::from_fn(|edge| tile.terrain_on_edge(tile_rotation, edge)),
        wildlife_destination: action.wildlife.map(AxialCoord::from_hex),
    })
}

fn selected_market_objects(
    draft: &DraftFactor,
    staged_market: &MarketSnapshot,
) -> Result<SelectedMarketObjects> {
    let (tile_slot, wildlife_slot) = draft.slots();
    let tile = staged_market.slots[usize::from(tile_slot)]
        .tile
        .clone()
        .ok_or_else(|| {
            R3Error::Invariant("selected tile slot is empty after prelude".to_owned())
        })?;
    let wildlife = staged_market.slots[usize::from(wildlife_slot)]
        .wildlife
        .ok_or_else(|| {
            R3Error::Invariant("selected wildlife slot is empty after prelude".to_owned())
        })?;
    Ok(SelectedMarketObjects {
        tile_slot,
        wildlife_slot,
        tile,
        wildlife,
    })
}

fn market_after_draft(
    staged_market: &MarketSnapshot,
    draft: &DraftFactor,
) -> Result<MarketSnapshot> {
    let mut after = staged_market.clone();
    let (tile_slot, wildlife_slot) = draft.slots();
    if after.slots[usize::from(tile_slot)].tile.take().is_none() {
        return Err(R3Error::Invariant(
            "draft removes an empty tile slot".to_owned(),
        ));
    }
    if after.slots[usize::from(wildlife_slot)]
        .wildlife
        .take()
        .is_none()
    {
        return Err(R3Error::Invariant(
            "draft removes an empty wildlife slot".to_owned(),
        ));
    }
    Ok(after)
}

fn supply_after_draft(
    staged: &SupplySnapshot,
    selected_wildlife: Wildlife,
    wildlife_placed: bool,
) -> Result<SupplySnapshot> {
    let mut after = staged.clone();
    if !wildlife_placed {
        let count = &mut after.wildlife_bag[selected_wildlife as usize];
        *count = count
            .checked_add(1)
            .ok_or_else(|| R3Error::Invariant("wildlife supply overflowed".to_owned()))?;
    }
    after.validate()?;
    Ok(after)
}

fn score_delta_between(before: ScoreBreakdown, after: ScoreBreakdown) -> ImmediateScoreDelta {
    ImmediateScoreDelta {
        habitat: std::array::from_fn(|index| {
            after.habitat[index] as i16 - before.habitat[index] as i16
        }),
        wildlife: std::array::from_fn(|index| {
            after.wildlife[index] as i16 - before.wildlife[index] as i16
        }),
        nature_tokens: after.nature_tokens as i16 - before.nature_tokens as i16,
        base_total: after.base_total as i16 - before.base_total as i16,
    }
}

fn score_delta_from_records(
    before_record: &PositionRecord,
    after_record: &PositionRecord,
) -> Result<ImmediateScoreDelta> {
    let cards = scoring_cards_from_codes(before_record.scoring_cards)?;
    if after_record.scoring_cards != before_record.scoring_cards {
        return Err(R3Error::Invariant(
            "an action changed the scoring-card configuration".to_owned(),
        ));
    }
    let before = score_board(&board_from_record(before_record, 0)?, cards);
    let after = score_board(&board_from_record(after_record, 0)?, cards);
    let habitat =
        std::array::from_fn(|index| after.habitat[index] as i16 - before.habitat[index] as i16);
    let wildlife =
        std::array::from_fn(|index| after.wildlife[index] as i16 - before.wildlife[index] as i16);
    let nature_tokens =
        i16::from(after_record.nature_tokens[0]) - i16::from(before_record.nature_tokens[0]);
    Ok(ImmediateScoreDelta {
        habitat,
        wildlife,
        nature_tokens,
        base_total: habitat.iter().sum::<i16>() + wildlife.iter().sum::<i16>() + nature_tokens,
    })
}

fn scoring_cards_from_codes(codes: [u8; 5]) -> Result<ScoringCards> {
    let variants = codes
        .map(|code| match code {
            0 => Ok(ScoringVariant::A),
            1 => Ok(ScoringVariant::B),
            2 => Ok(ScoringVariant::C),
            3 => Ok(ScoringVariant::D),
            _ => Err(R3Error::Invariant(format!(
                "invalid scoring-card code {code}"
            ))),
        })
        .into_iter()
        .collect::<Result<Vec<_>>>()?;
    Ok(ScoringCards {
        bear: variants[0],
        elk: variants[1],
        salmon: variants[2],
        hawk: variants[3],
        fox: variants[4],
    })
}

fn board_from_record(record: &PositionRecord, relative_seat: usize) -> Result<Board> {
    let tokens = board_map(record, relative_seat)?
        .into_values()
        .collect::<Vec<_>>();
    let mut pending = tokens.clone();
    let mut board = Board::empty();
    while !pending.is_empty() {
        let next = if board.tile_count() == 0 {
            Some(0)
        } else {
            pending.iter().position(|token| {
                token.coord.to_hex().is_ok_and(|coord| {
                    coord
                        .neighbors()
                        .into_iter()
                        .any(|neighbor| board.tile_at(neighbor).is_some())
                })
            })
        }
        .ok_or_else(|| R3Error::Invariant("public board entities are disconnected".to_owned()))?;
        let token = pending.remove(next);
        board
            .place_tile(
                token.coord.to_hex()?,
                token.tile.as_tile(),
                Rotation::new(token.rotation).ok_or_else(|| {
                    R3Error::Invariant("board token has an invalid rotation".to_owned())
                })?,
            )
            .map_err(|error| {
                R3Error::Invariant(format!("failed to reconstruct public board tile: {error}"))
            })?;
    }
    for token in tokens {
        if let Some(wildlife) = token.placed_wildlife {
            board
                .place_wildlife(token.coord.to_hex()?, wildlife)
                .map_err(|error| {
                    R3Error::Invariant(format!(
                        "failed to reconstruct public board wildlife: {error}"
                    ))
                })?;
        }
    }
    Ok(board)
}

fn write_active_board(record: &mut PositionRecord, board: &Board) {
    record.board_counts[0] = board.tile_count() as u8;
    record.nature_tokens[0] = board.nature_tokens();
    record.wildlife_counts[0] = [0; 5];
    record.habitat_sizes[0] = [0; 5];
    record.board_entities[0] = [[NONE; BOARD_ENTITY_SIZE]; MAX_BOARD_TILES];
    for wildlife in Wildlife::ALL {
        record.wildlife_counts[0][wildlife as usize] =
            board.wildlife_positions(wildlife).len() as u8;
    }
    for terrain in Terrain::ALL {
        record.habitat_sizes[0][terrain as usize] = board.largest_habitat(terrain);
    }
    let mut tiles = board.placed_tiles().collect::<Vec<_>>();
    tiles.sort_unstable_by_key(|(coord, _)| (coord.q, coord.r));
    for (index, (coord, placed)) in tiles.into_iter().enumerate() {
        record.board_entities[0][index] = [
            coord.q as u8,
            coord.r as u8,
            placed.tile.terrain_a as u8,
            placed.tile.terrain_b.map_or(NONE, |terrain| terrain as u8),
            placed.rotation.get(),
            placed.tile.wildlife.bits(),
            placed.wildlife.map_or(NONE, |wildlife| wildlife as u8),
            u8::from(placed.tile.keystone),
        ];
    }
}

fn write_market_snapshot(record: &mut PositionRecord, market: &MarketSnapshot) -> Result<()> {
    market.validate()?;
    for token in &market.slots {
        record.market_entities[usize::from(token.slot)] = market_token_to_entity(token)?;
    }
    Ok(())
}

fn validate_supply_delta_semantics(
    delta: &SupplyDelta,
    market_before: &MarketSnapshot,
    market_after: &MarketSnapshot,
    board: Option<&BoardObjectChanges>,
) -> Result<()> {
    let mut wildlife_delta = [0i16; 5];
    let mut archetype_delta = BTreeMap::<u16, i16>::new();
    for token in &market_before.slots {
        add_visible_market_token(token, 1, &mut wildlife_delta, &mut archetype_delta)?;
    }
    for token in &market_after.slots {
        add_visible_market_token(token, -1, &mut wildlife_delta, &mut archetype_delta)?;
    }
    if let Some(board) = board {
        for token in &board.removed {
            add_visible_board_token(token, 1, &mut wildlife_delta, &mut archetype_delta)?;
        }
        for token in &board.added {
            add_visible_board_token(token, -1, &mut wildlife_delta, &mut archetype_delta)?;
        }
        for update in &board.updated {
            add_visible_wildlife(update.before.placed_wildlife, 1, &mut wildlife_delta);
            add_visible_wildlife(update.after.placed_wildlife, -1, &mut wildlife_delta);
            if update.before.tile != update.after.tile {
                add_visible_tile(
                    &update.before.tile,
                    1,
                    &mut archetype_delta,
                    "updated board tile before",
                )?;
                add_visible_tile(
                    &update.after.tile,
                    -1,
                    &mut archetype_delta,
                    "updated board tile after",
                )?;
            }
        }
    }
    let archetype_deltas = archetype_delta
        .into_iter()
        .filter_map(|(archetype_id, delta)| {
            (delta != 0).then_some(SupplyCountDelta {
                archetype_id,
                delta,
            })
        })
        .collect::<Vec<_>>();
    let tile_delta = archetype_deltas
        .iter()
        .map(|delta| delta.delta)
        .sum::<i16>();
    if delta.wildlife_delta != wildlife_delta
        || delta.archetype_deltas != archetype_deltas
        || delta.unseen_tile_delta != tile_delta
        || delta.drawable_tile_delta != tile_delta
    {
        return Err(R3Error::Invariant(
            "public supply delta disagrees with visible market and board edits".to_owned(),
        ));
    }
    Ok(())
}

fn add_visible_market_token(
    token: &MarketSlotToken,
    supply_sign: i16,
    wildlife_delta: &mut [i16; 5],
    archetype_delta: &mut BTreeMap<u16, i16>,
) -> Result<()> {
    add_visible_wildlife(token.wildlife, supply_sign, wildlife_delta);
    if let Some(tile) = &token.tile {
        add_visible_tile(tile, supply_sign, archetype_delta, "market tile")?;
    }
    Ok(())
}

fn add_visible_board_token(
    token: &BoardTileToken,
    supply_sign: i16,
    wildlife_delta: &mut [i16; 5],
    archetype_delta: &mut BTreeMap<u16, i16>,
) -> Result<()> {
    add_visible_wildlife(token.placed_wildlife, supply_sign, wildlife_delta);
    add_visible_tile(
        &token.tile,
        supply_sign,
        archetype_delta,
        "changed board tile",
    )
}

fn add_visible_wildlife(
    wildlife: Option<Wildlife>,
    supply_sign: i16,
    wildlife_delta: &mut [i16; 5],
) {
    if let Some(wildlife) = wildlife {
        wildlife_delta[wildlife as usize] += supply_sign;
    }
}

fn add_visible_tile(
    tile: &TileSemantic,
    supply_sign: i16,
    archetype_delta: &mut BTreeMap<u16, i16>,
    label: &str,
) -> Result<()> {
    let archetype_id = tile.semantic_archetype_id.ok_or_else(|| {
        R3Error::Invariant(format!(
            "{label} is not part of the standard semantic supply catalog"
        ))
    })?;
    *archetype_delta.entry(archetype_id).or_default() += supply_sign;
    Ok(())
}

fn validate_board_action(edit: &ActionEdit) -> Result<()> {
    let changes = &edit.placement.board;
    if changes.added.len() != 1 || !changes.removed.is_empty() {
        return Err(R3Error::Invariant(
            "a complete action must add exactly one tile and remove none".to_owned(),
        ));
    }
    let added = &changes.added[0];
    if added.coord != edit.factors.tile_destination
        || added.tile != edit.selected.tile
        || added.rotation != edit.factors.tile_rotation
        || added.directed_edge_terrains != edit.factors.tile_directed_edges
    {
        return Err(R3Error::Invariant(
            "board addition disagrees with selected tile placement".to_owned(),
        ));
    }
    match edit.factors.wildlife_destination {
        None => {
            if added.placed_wildlife.is_some() || !changes.updated.is_empty() {
                return Err(R3Error::Invariant(
                    "discarded wildlife unexpectedly changed a board tile".to_owned(),
                ));
            }
        }
        Some(destination) if destination == edit.factors.tile_destination => {
            if added.placed_wildlife != Some(edit.selected.wildlife) || !changes.updated.is_empty()
            {
                return Err(R3Error::Invariant(
                    "wildlife on the new tile is not represented by the tile addition".to_owned(),
                ));
            }
        }
        Some(destination) => {
            if added.placed_wildlife.is_some() || changes.updated.len() != 1 {
                return Err(R3Error::Invariant(
                    "wildlife on an existing tile must be one exact board update".to_owned(),
                ));
            }
            let update = &changes.updated[0];
            if update.before.coord != destination
                || update.after.coord != destination
                || update.before.tile != update.after.tile
                || update.before.rotation != update.after.rotation
                || update.before.directed_edge_terrains != update.after.directed_edge_terrains
                || update.before.placed_wildlife.is_some()
                || update.after.placed_wildlife != Some(edit.selected.wildlife)
            {
                return Err(R3Error::Invariant(
                    "existing-tile wildlife update changes more than wildlife occupancy".to_owned(),
                ));
            }
        }
    }
    Ok(())
}

fn validate_player_changes(edit: &ActionEdit) -> Result<()> {
    let before = &edit.prelude.active_player_before;
    let staged = &edit.prelude.active_player_after;
    if before.occupied_count != staged.occupied_count
        || before.wildlife_counts != staged.wildlife_counts
        || before.largest_habitats != staged.largest_habitats
    {
        return Err(R3Error::Invariant(
            "market prelude changed public board geometry".to_owned(),
        ));
    }
    let wipe_count = i16::try_from(edit.factors.wildlife_wipe_masks.len())
        .map_err(|_| R3Error::Invariant("wildlife wipe sequence is too long".to_owned()))?;
    if i16::from(staged.nature_tokens) != i16::from(before.nature_tokens) - wipe_count {
        return Err(R3Error::Invariant(
            "market-prelude Nature Token delta disagrees with paid wipes".to_owned(),
        ));
    }

    let after = &edit.placement.active_player_after;
    if after.occupied_count
        != staged.occupied_count.checked_add(1).ok_or_else(|| {
            R3Error::Invariant("active-player occupied count overflowed".to_owned())
        })?
        || after
            .largest_habitats
            .iter()
            .zip(staged.largest_habitats)
            .any(|(after, before)| *after < before)
    {
        return Err(R3Error::Invariant(
            "placement summary does not represent one nonshrinking tile addition".to_owned(),
        ));
    }
    let mut expected_wildlife = staged.wildlife_counts;
    if edit.factors.wildlife_destination.is_some() {
        expected_wildlife[edit.selected.wildlife as usize] = expected_wildlife
            [edit.selected.wildlife as usize]
            .checked_add(1)
            .ok_or_else(|| R3Error::Invariant("wildlife count overflowed".to_owned()))?;
    }
    if after.wildlife_counts != expected_wildlife {
        return Err(R3Error::Invariant(
            "placement wildlife counts disagree with the selected destination".to_owned(),
        ));
    }
    let independent_cost = i16::from(matches!(
        edit.factors.draft,
        DraftFactor::Independent { .. }
    ));
    let keystone_grant = match edit.factors.wildlife_destination {
        None => 0,
        Some(destination) if destination == edit.factors.tile_destination => {
            i16::from(edit.placement.board.added[0].tile.keystone)
        }
        Some(_) => i16::from(edit.placement.board.updated[0].after.tile.keystone),
    };
    if i16::from(after.nature_tokens)
        != i16::from(staged.nature_tokens) - independent_cost + keystone_grant
    {
        return Err(R3Error::Invariant(
            "placement Nature Token delta disagrees with draft cost and keystone grant".to_owned(),
        ));
    }

    let expected_habitat_delta = std::array::from_fn(|index| {
        i16::from(after.largest_habitats[index]) - i16::from(before.largest_habitats[index])
    });
    let expected_nature_delta = i16::from(after.nature_tokens) - i16::from(before.nature_tokens);
    if edit.score_delta.habitat != expected_habitat_delta
        || edit.score_delta.nature_tokens != expected_nature_delta
        || edit.score_delta.base_total
            != edit.score_delta.habitat.iter().sum::<i16>()
                + edit.score_delta.wildlife.iter().sum::<i16>()
                + edit.score_delta.nature_tokens
    {
        return Err(R3Error::Invariant(
            "score anatomy disagrees with public summaries or component sum".to_owned(),
        ));
    }
    Ok(())
}

fn validate_world_edit_order(edit: &ActionEdit) -> Result<()> {
    if !strictly_ordered_by(&edit.placement.board.added, |token| token.coord)
        || !strictly_ordered_by(&edit.placement.board.removed, |token| token.coord)
        || !strictly_ordered_by(&edit.placement.board.updated, |update| update.before.coord)
        || !strictly_ordered_by(&edit.frontier.added, |token| AxialCoord::from(token.coord))
        || !strictly_ordered_by(&edit.frontier.removed, |token| {
            AxialCoord::from(token.coord)
        })
        || !strictly_ordered_by(&edit.frontier.updated, |update| {
            AxialCoord::from(update.before.coord)
        })
        || !strictly_ordered_by(&edit.components.added, |token| token.object_key)
        || !strictly_ordered_by(&edit.components.removed, |token| token.object_key)
        || !strictly_ordered_by(&edit.components.updated, |update| update.before.object_key)
        || !strictly_ordered_by(&edit.motifs.added, |token| token.coord)
        || !strictly_ordered_by(&edit.motifs.removed, |token| token.coord)
        || !strictly_ordered_by(&edit.motifs.updated, |update| update.before.coord)
    {
        return Err(R3Error::Invariant(
            "world edit collections are duplicated or noncanonical".to_owned(),
        ));
    }
    for token in edit
        .placement
        .board
        .added
        .iter()
        .chain(&edit.placement.board.removed)
        .chain(
            edit.placement
                .board
                .updated
                .iter()
                .flat_map(|update| [&update.before, &update.after]),
        )
    {
        token.to_entity()?;
    }
    for component in edit
        .components
        .added
        .iter()
        .chain(&edit.components.removed)
        .chain(
            edit.components
                .updated
                .iter()
                .flat_map(|update| [&update.before, &update.after]),
        )
    {
        validate_component_object(component)?;
    }
    Ok(())
}

fn validate_canonical_edit_order(edit: &CanonicalGlobalEdit) -> Result<()> {
    if !strictly_ordered_by(&edit.board_added, |token| token.offset)
        || !strictly_ordered_by(&edit.board_removed, |token| token.offset)
        || !strictly_ordered_by(&edit.frontier_added, |token| token.offset)
        || !strictly_ordered_by(&edit.frontier_removed, |token| token.offset)
        || !strictly_ordered_by(&edit.motifs_added, |token| token.offset)
        || !strictly_ordered_by(&edit.motifs_removed, |token| token.offset)
    {
        return Err(R3Error::Invariant(
            "canonical spatial edit collections are duplicated or noncanonical".to_owned(),
        ));
    }
    validate_postcard_order(&edit.board_updated, "canonical board updates")?;
    validate_postcard_order(&edit.frontier_updated, "canonical frontier updates")?;
    validate_postcard_order(&edit.components_added, "canonical added components")?;
    validate_postcard_order(&edit.components_removed, "canonical removed components")?;
    validate_postcard_order(&edit.components_updated, "canonical component updates")?;
    validate_postcard_order(&edit.motifs_updated, "canonical motif updates")?;
    for component in edit
        .components_added
        .iter()
        .chain(&edit.components_removed)
        .chain(
            edit.components_updated
                .iter()
                .flat_map(|update| [&update.before, &update.after]),
        )
    {
        if component.member_count as usize != component.members.len()
            || !strictly_ordered(&component.members)
        {
            return Err(R3Error::Invariant(
                "canonical component membership is inconsistent".to_owned(),
            ));
        }
    }
    Ok(())
}

fn validate_component_object(component: &ComponentObject) -> Result<()> {
    if component.member_count as usize != component.members.len()
        || !strictly_ordered(&component.members)
        || component.object_key
            != component_key(
                component.relative_seat,
                component.terrain,
                &component.members,
            )
    {
        return Err(R3Error::Invariant(
            "component object identity or membership is inconsistent".to_owned(),
        ));
    }
    Ok(())
}

fn validate_global_reference_order(references: &GlobalObjectReferences) -> Result<()> {
    if !strictly_ordered(&references.before_component_keys)
        || !strictly_ordered(&references.after_component_keys)
        || !strictly_ordered(&references.frontier_coords)
        || !strictly_ordered(&references.motif_coords)
        || !strictly_ordered(&references.market_slots)
        || !strictly_ordered(&references.supply_archetype_ids)
    {
        return Err(R3Error::Invariant(
            "global object references are duplicated or noncanonical".to_owned(),
        ));
    }
    Ok(())
}

fn strictly_ordered<T: Ord>(values: &[T]) -> bool {
    values.windows(2).all(|pair| pair[0] < pair[1])
}

fn strictly_ordered_by<T, K: Ord>(values: &[T], key: impl Fn(&T) -> K) -> bool {
    values.windows(2).all(|pair| key(&pair[0]) < key(&pair[1]))
}

fn validate_postcard_order<T: Serialize>(values: &[T], label: &str) -> Result<()> {
    let mut previous: Option<Vec<u8>> = None;
    for value in values {
        let bytes = postcard::to_allocvec(value)?;
        if previous.as_ref().is_some_and(|prior| prior >= &bytes) {
            return Err(R3Error::Invariant(format!(
                "{label} are duplicated or noncanonical"
            )));
        }
        previous = Some(bytes);
    }
    Ok(())
}

fn validate_sorted_unique_edits(edits: &[MarketSlotEdit]) -> Result<()> {
    if edits.windows(2).any(|pair| pair[0].slot >= pair[1].slot)
        || edits.iter().any(|edit| {
            edit.slot >= 4 || edit.before.slot != edit.slot || edit.after.slot != edit.slot
        })
    {
        return Err(R3Error::Invariant(
            "market edits are not strictly ordered canonical slots".to_owned(),
        ));
    }
    Ok(())
}

fn board_changes(before: &PositionRecord, after: &PositionRecord) -> Result<BoardObjectChanges> {
    for seat in 1..usize::from(before.player_count) {
        if before.board_counts[seat] != after.board_counts[seat]
            || before.nature_tokens[seat] != after.nature_tokens[seat]
            || before.wildlife_counts[seat] != after.wildlife_counts[seat]
            || before.habitat_sizes[seat] != after.habitat_sizes[seat]
            || before.board_entities[seat] != after.board_entities[seat]
        {
            return Err(R3Error::Invariant(
                "an action changed a nonacting relative board".to_owned(),
            ));
        }
    }
    let before = board_map(before, 0)?;
    let after = board_map(after, 0)?;
    let mut changes = BoardObjectChanges::default();
    for (coord, token) in &before {
        match after.get(coord) {
            None => changes.removed.push(token.clone()),
            Some(updated) if updated != token => changes.updated.push(ObjectUpdate {
                before: token.clone(),
                after: updated.clone(),
            }),
            Some(_) => {}
        }
    }
    for (coord, token) in &after {
        if !before.contains_key(coord) {
            changes.added.push(token.clone());
        }
    }
    Ok(changes)
}

fn board_map(
    record: &PositionRecord,
    relative_seat: usize,
) -> Result<BTreeMap<AxialCoord, BoardTileToken>> {
    let mut result = BTreeMap::new();
    for row in 0..usize::from(record.board_counts[relative_seat]) {
        let token = BoardTileToken::from_entity(record.board_entities[relative_seat][row])?;
        if result.insert(token.coord, token).is_some() {
            return Err(R3Error::Invariant(
                "board record contains a duplicate coordinate".to_owned(),
            ));
        }
    }
    Ok(result)
}

fn apply_board_changes(record: &mut PositionRecord, changes: &BoardObjectChanges) -> Result<()> {
    let mut board = board_map(record, 0)?;
    for removed in &changes.removed {
        match board.remove(&removed.coord) {
            Some(actual) if actual == *removed => {}
            _ => {
                return Err(R3Error::Invariant(
                    "board removal precondition mismatch".to_owned(),
                ));
            }
        }
    }
    for update in &changes.updated {
        match board.get(&update.before.coord) {
            Some(actual) if *actual == update.before => {}
            _ => {
                return Err(R3Error::Invariant(
                    "board update precondition mismatch".to_owned(),
                ));
            }
        }
        if update.before.coord != update.after.coord {
            return Err(R3Error::Invariant(
                "board update cannot move a tile".to_owned(),
            ));
        }
        board.insert(update.after.coord, update.after.clone());
    }
    for added in &changes.added {
        if board.insert(added.coord, added.clone()).is_some() {
            return Err(R3Error::Invariant(
                "board addition overwrites an occupied coordinate".to_owned(),
            ));
        }
    }
    if board.len() > MAX_BOARD_TILES {
        return Err(R3Error::Invariant(
            "board edit exceeds the 23-tile rules limit".to_owned(),
        ));
    }
    record.board_entities[0] = [[NONE; BOARD_ENTITY_SIZE]; MAX_BOARD_TILES];
    for (row, token) in board.values().enumerate() {
        record.board_entities[0][row] = token.to_entity()?;
    }
    Ok(())
}

fn frontier_changes(
    before: &SparsePublicState,
    after: &SparsePublicState,
) -> Result<FrontierChanges> {
    let before_components = component_lookup(before);
    let after_components = component_lookup(after);
    let before = before
        .legal_frontier
        .iter()
        .filter(|token| token.relative_seat == 0)
        .map(|token| (AxialCoord::from(token.coord), token.clone()))
        .collect::<BTreeMap<_, _>>();
    let after = after
        .legal_frontier
        .iter()
        .filter(|token| token.relative_seat == 0)
        .map(|token| (AxialCoord::from(token.coord), token.clone()))
        .collect::<BTreeMap<_, _>>();
    let mut changes = FrontierChanges::default();
    for (coord, token) in &before {
        match after.get(coord) {
            None => changes.removed.push(token.clone()),
            Some(updated)
                if canonical_frontier_token(
                    token,
                    &before_components,
                    AxialCoord::ORIGIN,
                    D6Transform::IDENTITY,
                )? != canonical_frontier_token(
                    updated,
                    &after_components,
                    AxialCoord::ORIGIN,
                    D6Transform::IDENTITY,
                )? =>
            {
                changes.updated.push(ObjectUpdate {
                    before: token.clone(),
                    after: updated.clone(),
                });
            }
            Some(_) => {}
        }
    }
    for (coord, token) in &after {
        if !before.contains_key(coord) {
            changes.added.push(token.clone());
        }
    }
    Ok(changes)
}

fn component_changes(before: &SparsePublicState, after: &SparsePublicState) -> ComponentChanges {
    let before = before
        .habitat_components
        .iter()
        .filter(|token| token.relative_seat == 0)
        .map(ComponentObject::from_r2)
        .map(|token| (token.object_key, token))
        .collect::<BTreeMap<_, _>>();
    let after = after
        .habitat_components
        .iter()
        .filter(|token| token.relative_seat == 0)
        .map(ComponentObject::from_r2)
        .map(|token| (token.object_key, token))
        .collect::<BTreeMap<_, _>>();
    let mut changes = ComponentChanges::default();
    for (key, token) in &before {
        match after.get(key) {
            None => changes.removed.push(token.clone()),
            Some(updated) if updated != token => changes.updated.push(ObjectUpdate {
                before: token.clone(),
                after: updated.clone(),
            }),
            Some(_) => {}
        }
    }
    for (key, token) in &after {
        if !before.contains_key(key) {
            changes.added.push(token.clone());
        }
    }
    changes
}

fn motif_changes(before: &SparsePublicState, after: &SparsePublicState) -> MotifChanges {
    let before = before
        .wildlife_motifs
        .iter()
        .filter(|token| token.relative_seat == 0)
        .map(WildlifeMotifObject::from_r2)
        .map(|token| (token.coord, token))
        .collect::<BTreeMap<_, _>>();
    let after = after
        .wildlife_motifs
        .iter()
        .filter(|token| token.relative_seat == 0)
        .map(WildlifeMotifObject::from_r2)
        .map(|token| (token.coord, token))
        .collect::<BTreeMap<_, _>>();
    let mut changes = MotifChanges::default();
    for (coord, token) in &before {
        match after.get(coord) {
            None => changes.removed.push(token.clone()),
            Some(updated) if updated != token => changes.updated.push(ObjectUpdate {
                before: token.clone(),
                after: updated.clone(),
            }),
            Some(_) => {}
        }
    }
    for (coord, token) in &after {
        if !before.contains_key(coord) {
            changes.added.push(token.clone());
        }
    }
    changes
}

fn global_references(
    prelude: &PreludeEdit,
    placement: &PlacementEdit,
    frontier: &FrontierChanges,
    components: &ComponentChanges,
    motifs: &MotifChanges,
) -> GlobalObjectReferences {
    let mut before_component_keys = BTreeSet::new();
    let mut after_component_keys = BTreeSet::new();
    for component in &components.removed {
        before_component_keys.insert(component.object_key);
    }
    for update in &components.updated {
        before_component_keys.insert(update.before.object_key);
        after_component_keys.insert(update.after.object_key);
    }
    for component in &components.added {
        after_component_keys.insert(component.object_key);
    }
    let frontier_coords = frontier
        .added
        .iter()
        .chain(&frontier.removed)
        .map(|token| AxialCoord::from(token.coord))
        .chain(frontier.updated.iter().flat_map(|update| {
            [
                AxialCoord::from(update.before.coord),
                AxialCoord::from(update.after.coord),
            ]
        }))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let motif_coords = motifs
        .added
        .iter()
        .chain(&motifs.removed)
        .map(|token| token.coord)
        .chain(
            motifs
                .updated
                .iter()
                .flat_map(|update| [update.before.coord, update.after.coord]),
        )
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let market_slots = prelude
        .market_edits
        .iter()
        .chain(&placement.market_edits)
        .map(|edit| edit.slot)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let supply_archetype_ids = prelude
        .supply
        .archetype_deltas
        .iter()
        .chain(&placement.supply.archetype_deltas)
        .map(|delta| delta.archetype_id)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    GlobalObjectReferences {
        before_component_keys: before_component_keys.into_iter().collect(),
        after_component_keys: after_component_keys.into_iter().collect(),
        frontier_coords,
        motif_coords,
        market_slots,
        supply_archetype_ids,
    }
}

fn radius_coverage(
    factors: &ActionFactors,
    frontier: &FrontierChanges,
    motifs: &MotifChanges,
) -> [CoverageByRadius; 3] {
    let center = factors.tile_destination;
    let mut changed = BTreeSet::from([center]);
    if let Some(wildlife) = factors.wildlife_destination {
        changed.insert(wildlife);
    }
    changed.extend(
        frontier
            .added
            .iter()
            .chain(&frontier.removed)
            .map(|token| AxialCoord::from(token.coord)),
    );
    for update in &frontier.updated {
        changed.insert(update.before.coord.into());
        changed.insert(update.after.coord.into());
    }
    changed.extend(
        motifs
            .added
            .iter()
            .chain(&motifs.removed)
            .map(|token| token.coord),
    );
    for update in &motifs.updated {
        changed.insert(update.before.coord);
        changed.insert(update.after.coord);
    }
    std::array::from_fn(|index| {
        let radius = index as u8 + 1;
        let covered = changed
            .iter()
            .filter(|coord| coord.relative_to(center).distance() <= u16::from(radius))
            .count() as u16;
        CoverageByRadius {
            radius,
            changed_coordinate_count: changed.len() as u16,
            covered_coordinate_count: covered,
            complete: covered == changed.len() as u16,
        }
    })
}

#[allow(clippy::too_many_arguments)]
fn canonical_action_view(
    before: &SparsePublicState,
    after_components: &BTreeMap<u16, ComponentObject>,
    factors: &ActionFactors,
    selected: &SelectedMarketObjects,
    board: &BoardObjectChanges,
    frontier: &FrontierChanges,
    components: &ComponentChanges,
    motifs: &MotifChanges,
) -> Result<CanonicalActionView> {
    let tile = selected.tile.as_tile();
    let rotation = Rotation::new(factors.tile_rotation)
        .ok_or_else(|| R3Error::Invariant("invalid action rotation".to_owned()))?;
    let mut candidates = Vec::new();
    for transform in D6Transform::ALL {
        if transform.transform_tile_rotation(tile, rotation) != Rotation::ZERO {
            continue;
        }
        candidates.push(canonical_action_view_for_transform(
            before,
            after_components,
            factors,
            selected,
            board,
            frontier,
            components,
            motifs,
            transform,
        )?);
    }
    candidates
        .into_iter()
        .min_by_key(|candidate| postcard::to_allocvec(candidate).unwrap())
        .ok_or_else(|| R3Error::Invariant("no canonical action frame exists".to_owned()))
}

#[allow(clippy::too_many_arguments)]
fn canonical_action_view_for_transform(
    before: &SparsePublicState,
    after_components: &BTreeMap<u16, ComponentObject>,
    factors: &ActionFactors,
    selected: &SelectedMarketObjects,
    board: &BoardObjectChanges,
    frontier: &FrontierChanges,
    components: &ComponentChanges,
    motifs: &MotifChanges,
    transform: D6Transform,
) -> Result<CanonicalActionView> {
    let center = factors.tile_destination;
    let before_components = component_lookup(before);
    let local_patch = canonical_local_patch(before, center, transform)?;
    let selected_tile = canonical_board_token(
        &BoardTileToken {
            coord: center,
            tile: selected.tile.clone(),
            rotation: factors.tile_rotation,
            directed_edge_terrains: factors.tile_directed_edges,
            placed_wildlife: factors
                .wildlife_destination
                .filter(|coord| *coord == center)
                .map(|_| selected.wildlife),
        },
        center,
        transform,
    )?;
    let wildlife_destination_offset = factors
        .wildlife_destination
        .map(|coord| transform_relative(coord, center, transform))
        .transpose()?;
    let mut global_edit = CanonicalGlobalEdit {
        board_added: canonical_board_list(&board.added, center, transform)?,
        board_removed: canonical_board_list(&board.removed, center, transform)?,
        board_updated: board
            .updated
            .iter()
            .map(|update| {
                Ok(ObjectUpdate {
                    before: canonical_board_token(&update.before, center, transform)?,
                    after: canonical_board_token(&update.after, center, transform)?,
                })
            })
            .collect::<Result<_>>()?,
        frontier_added: canonical_frontier_list(
            &frontier.added,
            after_components,
            center,
            transform,
        )?,
        frontier_removed: canonical_frontier_list(
            &frontier.removed,
            &before_components,
            center,
            transform,
        )?,
        frontier_updated: frontier
            .updated
            .iter()
            .map(|update| {
                Ok(ObjectUpdate {
                    before: canonical_frontier_token(
                        &update.before,
                        &before_components,
                        center,
                        transform,
                    )?,
                    after: canonical_frontier_token(
                        &update.after,
                        after_components,
                        center,
                        transform,
                    )?,
                })
            })
            .collect::<Result<_>>()?,
        components_added: canonical_component_list(&components.added, center, transform)?,
        components_removed: canonical_component_list(&components.removed, center, transform)?,
        components_updated: components
            .updated
            .iter()
            .map(|update| {
                Ok(ObjectUpdate {
                    before: canonical_component(&update.before, center, transform)?,
                    after: canonical_component(&update.after, center, transform)?,
                })
            })
            .collect::<Result<_>>()?,
        motifs_added: canonical_motif_list(&motifs.added, center, transform)?,
        motifs_removed: canonical_motif_list(&motifs.removed, center, transform)?,
        motifs_updated: motifs
            .updated
            .iter()
            .map(|update| {
                Ok(ObjectUpdate {
                    before: canonical_motif(&update.before, center, transform)?,
                    after: canonical_motif(&update.after, center, transform)?,
                })
            })
            .collect::<Result<_>>()?,
    };
    sort_canonical_updates(&mut global_edit);
    Ok(CanonicalActionView {
        local_patch,
        selected_tile,
        wildlife_destination_offset,
        global_edit,
    })
}

fn sort_canonical_updates(edit: &mut CanonicalGlobalEdit) {
    sort_by_postcard(&mut edit.board_updated);
    sort_by_postcard(&mut edit.frontier_updated);
    sort_by_postcard(&mut edit.components_updated);
    sort_by_postcard(&mut edit.motifs_updated);
}

fn sort_by_postcard<T: Serialize>(values: &mut [T]) {
    values.sort_unstable_by_key(|value| {
        postcard::to_allocvec(value).expect("canonical edit serialization cannot fail")
    });
}

fn canonical_local_patch(
    state: &SparsePublicState,
    center: AxialCoord,
    transform: D6Transform,
) -> Result<CanonicalLocalPatch> {
    let occupied = state
        .occupied_tiles
        .iter()
        .filter(|token| token.relative_seat == 0)
        .map(|token| {
            let tile = TileSemantic::new(
                token.terrain_a,
                token.terrain_b,
                token.wildlife_eligibility,
                token.keystone,
            );
            (
                AxialCoord::from(token.coord),
                BoardTileToken {
                    coord: token.coord.into(),
                    tile,
                    rotation: token.rotation.get(),
                    directed_edge_terrains: token.directed_edge_terrains,
                    placed_wildlife: token.placed_wildlife,
                },
            )
        })
        .collect::<BTreeMap<_, _>>();
    let frontier = state
        .legal_frontier
        .iter()
        .filter(|token| token.relative_seat == 0)
        .map(|token| AxialCoord::from(token.coord))
        .collect::<BTreeSet<_>>();
    let mut cells = Vec::with_capacity(37);
    for q in -i16::from(LOCAL_PATCH_MAX_RADIUS)..=i16::from(LOCAL_PATCH_MAX_RADIUS) {
        for r in -i16::from(LOCAL_PATCH_MAX_RADIUS)..=i16::from(LOCAL_PATCH_MAX_RADIUS) {
            let offset = AxialCoord::new(q, r);
            if offset.distance() > u16::from(LOCAL_PATCH_MAX_RADIUS) {
                continue;
            }
            let world = AxialCoord::new(center.q + q, center.r + r);
            let inside_rules_grid = world.to_hex()?.to_index().is_some();
            let canonical_offset = offset.transformed_offset(transform)?;
            let occupied = occupied
                .get(&world)
                .map(|token| canonical_board_token(token, center, transform))
                .transpose()?;
            cells.push(LocalPatchCell {
                offset: canonical_offset,
                inside_rules_grid,
                frontier: frontier.contains(&world),
                occupied,
            });
        }
    }
    cells.sort_unstable_by_key(|cell| (cell.offset.q, cell.offset.r));
    Ok(CanonicalLocalPatch {
        radius: LOCAL_PATCH_MAX_RADIUS,
        cells,
    })
}

fn canonical_board_list(
    tokens: &[BoardTileToken],
    center: AxialCoord,
    transform: D6Transform,
) -> Result<Vec<CanonicalBoardToken>> {
    let mut result = tokens
        .iter()
        .map(|token| canonical_board_token(token, center, transform))
        .collect::<Result<Vec<_>>>()?;
    result.sort_unstable_by_key(|token| (token.offset.q, token.offset.r));
    Ok(result)
}

fn canonical_board_token(
    token: &BoardTileToken,
    center: AxialCoord,
    transform: D6Transform,
) -> Result<CanonicalBoardToken> {
    let source_rotation = Rotation::new(token.rotation)
        .ok_or_else(|| R3Error::Invariant("invalid board-token rotation".to_owned()))?;
    let target_rotation = transform.transform_tile_rotation(token.tile.as_tile(), source_rotation);
    let mut directed_edge_terrains = [Terrain::Mountain; 6];
    for edge in 0..6 {
        directed_edge_terrains[transform.transform_edge(edge)?] =
            token.directed_edge_terrains[edge];
    }
    Ok(CanonicalBoardToken {
        offset: transform_relative(token.coord, center, transform)?,
        tile: token.tile.clone(),
        rotation: target_rotation.get(),
        directed_edge_terrains,
        placed_wildlife: token.placed_wildlife,
    })
}

fn component_lookup(state: &SparsePublicState) -> BTreeMap<u16, ComponentObject> {
    state
        .habitat_components
        .iter()
        .filter(|token| token.relative_seat == 0)
        .map(|token| (token.component_id, ComponentObject::from_r2(token)))
        .collect()
}

fn canonical_component_list(
    tokens: &[ComponentObject],
    center: AxialCoord,
    transform: D6Transform,
) -> Result<Vec<CanonicalComponentObject>> {
    let mut result = tokens
        .iter()
        .map(|token| canonical_component(token, center, transform))
        .collect::<Result<Vec<_>>>()?;
    result.sort_unstable_by_key(|token| {
        postcard::to_allocvec(token).expect("canonical component serialization cannot fail")
    });
    Ok(result)
}

fn canonical_component(
    token: &ComponentObject,
    center: AxialCoord,
    transform: D6Transform,
) -> Result<CanonicalComponentObject> {
    let mut members = token
        .members
        .iter()
        .map(|member| transform_relative(*member, center, transform))
        .collect::<Result<Vec<_>>>()?;
    members.sort_unstable();
    Ok(CanonicalComponentObject {
        terrain: token.terrain,
        members,
        member_count: token.member_count,
        matching_internal_edge_count: token.matching_internal_edge_count,
        open_boundary_edge_count: token.open_boundary_edge_count,
        frontier_contact_count: token.frontier_contact_count,
    })
}

fn canonical_component_identity(
    token: &ComponentObject,
    center: AxialCoord,
    transform: D6Transform,
) -> Result<[u8; 32]> {
    Ok(*blake3::hash(&postcard::to_allocvec(&canonical_component(
        token, center, transform,
    )?)?)
    .as_bytes())
}

fn canonical_frontier_list(
    tokens: &[FrontierToken],
    components: &BTreeMap<u16, ComponentObject>,
    center: AxialCoord,
    transform: D6Transform,
) -> Result<Vec<CanonicalFrontierToken>> {
    let mut result = tokens
        .iter()
        .map(|token| canonical_frontier_token(token, components, center, transform))
        .collect::<Result<Vec<_>>>()?;
    result.sort_unstable_by_key(|token| (token.offset.q, token.offset.r));
    Ok(result)
}

fn canonical_frontier_token(
    token: &FrontierToken,
    components: &BTreeMap<u16, ComponentObject>,
    center: AxialCoord,
    transform: D6Transform,
) -> Result<CanonicalFrontierToken> {
    let mut neighbor_presence_bits = 0u8;
    let mut neighbor_facing_terrains = [None; 6];
    for edge in 0..6 {
        let target = transform.transform_edge(edge)?;
        if token.neighbor_presence_bits & (1 << edge) != 0 {
            neighbor_presence_bits |= 1 << target;
        }
        neighbor_facing_terrains[target] = token.neighbor_facing_terrains[edge];
    }
    let mut touched_habitat_components = token
        .touched_habitat_components
        .iter()
        .map(|touch| {
            let component = components.get(&touch.component_id).ok_or_else(|| {
                R3Error::Invariant(format!(
                    "frontier references missing component {}",
                    touch.component_id
                ))
            })?;
            Ok(CanonicalFrontierTouch {
                terrain: touch.terrain,
                component_key: canonical_component_identity(component, center, transform)?,
                component_size: touch.component_size,
                contact_edge_bits: transform_edge_bits(touch.contact_edge_bits, transform)?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    touched_habitat_components.sort_unstable_by_key(|touch| {
        (
            touch.terrain as u8,
            touch.component_key,
            touch.contact_edge_bits,
        )
    });
    Ok(CanonicalFrontierToken {
        offset: transform_relative(token.coord.into(), center, transform)?,
        neighbor_presence_bits,
        neighbor_facing_terrains,
        adjacent_wildlife_counts: token.adjacent_wildlife_counts,
        occupied_neighbor_runs: token.occupied_neighbor_runs,
        opposite_neighbor_pair_bits: opposite_pair_bits(neighbor_presence_bits),
        touched_habitat_components,
        resulting_size_by_terrain: token.resulting_size_by_terrain,
        habitat_bridge_terrain_bits: token.habitat_bridge_terrain_bits,
        repeated_component_contact_terrain_bits: token.repeated_component_contact_terrain_bits,
    })
}

fn canonical_motif_list(
    tokens: &[WildlifeMotifObject],
    center: AxialCoord,
    transform: D6Transform,
) -> Result<Vec<CanonicalMotifObject>> {
    let mut result = tokens
        .iter()
        .map(|token| canonical_motif(token, center, transform))
        .collect::<Result<Vec<_>>>()?;
    result.sort_unstable_by_key(|token| (token.offset.q, token.offset.r));
    Ok(result)
}

fn canonical_motif(
    token: &WildlifeMotifObject,
    center: AxialCoord,
    transform: D6Transform,
) -> Result<CanonicalMotifObject> {
    let mut neighbor_wildlife = [None; 6];
    for edge in 0..6 {
        neighbor_wildlife[transform.transform_edge(edge)?] = token.neighbor_wildlife[edge];
    }
    Ok(CanonicalMotifObject {
        offset: transform_relative(token.coord, center, transform)?,
        wildlife: token.wildlife,
        neighbor_wildlife,
        adjacent_wildlife_counts: token.adjacent_wildlife_counts,
        same_species_neighbor_bits: transform_edge_bits(
            token.same_species_neighbor_bits,
            transform,
        )?,
    })
}

fn transform_relative(
    coord: AxialCoord,
    center: AxialCoord,
    transform: D6Transform,
) -> Result<AxialCoord> {
    coord.relative_to(center).transformed_offset(transform)
}

fn transform_edge_bits(bits: u8, transform: D6Transform) -> Result<u8> {
    let mut transformed = 0u8;
    for edge in 0..6 {
        if bits & (1 << edge) != 0 {
            transformed |= 1 << transform.transform_edge(edge)?;
        }
    }
    Ok(transformed)
}

fn opposite_pair_bits(neighbor_bits: u8) -> u8 {
    let mut result = 0u8;
    for edge in 0..3 {
        if neighbor_bits & (1 << edge) != 0 && neighbor_bits & (1 << (edge + 3)) != 0 {
            result |= 1 << edge;
        }
    }
    result
}

fn sparse_geometry_afterstate(
    after: &PositionRecord,
    before: &PositionRecord,
) -> Result<SparsePublicState> {
    let mut geometry = after.clone();
    geometry.market_entities = before.market_entities;
    SparsePublicState::from_position_record(&geometry, None).map_err(Into::into)
}

fn apply_market_edits(record: &mut PositionRecord, edits: &[MarketSlotEdit]) -> Result<()> {
    validate_sorted_unique_edits(edits)?;
    for edit in edits {
        let slot = usize::from(edit.slot);
        let actual = market_token_from_entity(edit.slot, record.market_entities[slot])?;
        if actual != edit.before {
            return Err(R3Error::Invariant(
                "market edit precondition mismatch".to_owned(),
            ));
        }
        record.market_entities[slot] = market_token_to_entity(&edit.after)?;
    }
    Ok(())
}

fn market_snapshot_from_record(record: &PositionRecord) -> Result<MarketSnapshot> {
    let slots = (0..4)
        .map(|slot| market_token_from_entity(slot as u8, record.market_entities[slot]))
        .collect::<Result<Vec<_>>>()?
        .try_into()
        .map_err(|_| R3Error::Invariant("market snapshot must contain four slots".to_owned()))?;
    Ok(MarketSnapshot { slots })
}

fn market_token_from_entity(slot: u8, entity: [u8; MARKET_ENTITY_SIZE]) -> Result<MarketSlotToken> {
    if entity == [NONE; MARKET_ENTITY_SIZE] {
        return Ok(MarketSlotToken {
            slot,
            tile: None,
            wildlife: None,
        });
    }
    if entity[5..] != [0, 0, 0] {
        return Err(R3Error::Invariant(
            "market entity reserved bytes are nonzero".to_owned(),
        ));
    }
    let tile = if entity[0] == NONE {
        if entity[1] != NONE || entity[2] != 0 || entity[4] != 0 {
            return Err(R3Error::Invariant(
                "market entity has partial tile semantics".to_owned(),
            ));
        }
        None
    } else {
        let mask = WildlifeMask::from_bits(entity[2]);
        if mask.bits() != entity[2] {
            return Err(R3Error::Invariant(
                "market tile wildlife mask is invalid".to_owned(),
            ));
        }
        let keystone = match entity[4] {
            0 => false,
            1 => true,
            _ => {
                return Err(R3Error::Invariant(
                    "market tile keystone flag is invalid".to_owned(),
                ));
            }
        };
        Some(TileSemantic::new(
            terrain(entity[0])?,
            optional_terrain(entity[1])?,
            mask,
            keystone,
        ))
    };
    Ok(MarketSlotToken {
        slot,
        tile,
        wildlife: optional_wildlife(entity[3])?,
    })
}

fn market_token_to_entity(token: &MarketSlotToken) -> Result<[u8; MARKET_ENTITY_SIZE]> {
    if token.tile.is_none() && token.wildlife.is_none() {
        return Ok([NONE; MARKET_ENTITY_SIZE]);
    }
    if let Some(tile) = &token.tile {
        tile.validate()?;
    }
    Ok([
        token
            .tile
            .as_ref()
            .map_or(NONE, |tile| tile.terrain_a as u8),
        token
            .tile
            .as_ref()
            .and_then(|tile| tile.terrain_b)
            .map_or(NONE, |terrain| terrain as u8),
        token
            .tile
            .as_ref()
            .map_or(0, |tile| tile.wildlife_eligibility.bits()),
        token.wildlife.map_or(NONE, |wildlife| wildlife as u8),
        token
            .tile
            .as_ref()
            .map_or(0, |tile| u8::from(tile.keystone)),
        0,
        0,
        0,
    ])
}

fn terrain(code: u8) -> Result<Terrain> {
    Terrain::ALL
        .get(usize::from(code))
        .copied()
        .ok_or_else(|| R3Error::Invariant(format!("invalid terrain code {code}")))
}

fn optional_terrain(code: u8) -> Result<Option<Terrain>> {
    if code == NONE {
        Ok(None)
    } else {
        terrain(code).map(Some)
    }
}

fn optional_wildlife(code: u8) -> Result<Option<Wildlife>> {
    if code == NONE {
        Ok(None)
    } else {
        Wildlife::ALL
            .get(usize::from(code))
            .copied()
            .map(Some)
            .ok_or_else(|| R3Error::Invariant(format!("invalid wildlife code {code}")))
    }
}

fn normalized_record_bytes(record: &PositionRecord) -> [u8; cascadia_data::RECORD_SIZE] {
    let mut normalized = record.clone();
    normalized.targets = [0; TARGET_DIM];
    normalized.to_bytes()
}

fn record_hash(record: &PositionRecord) -> [u8; 32] {
    *blake3::hash(&normalized_record_bytes(record)).as_bytes()
}

impl AppliedPublicState {
    pub fn canonical_record_hash(&self) -> [u8; 32] {
        record_hash(&self.record)
    }
}

impl PublicStateTrunk {
    pub fn public_record_hash(&self) -> Result<[u8; 32]> {
        Ok(record_hash(&self.public_record()?))
    }
}
