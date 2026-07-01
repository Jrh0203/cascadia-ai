use std::collections::{BTreeMap, BTreeSet};

use cascadia_game::{Terrain, Wildlife, WildlifeMask};

use crate::{
    ActionEdit, AxialCoord, CanonicalActionView, CanonicalBoardToken, CanonicalComponentObject,
    CanonicalFrontierToken, CanonicalFrontierTouch, CanonicalGlobalEdit, CanonicalLocalPatch,
    CanonicalMotifObject, LocalPatchCell, ObjectUpdate, R3Error, Result, TileSemantic,
};

pub const MLX_ACTION_ENCODING_SCHEMA_VERSION: u16 = 1;
pub const MLX_ACTION_TOKEN_PAYLOAD_WIDTH: usize = 64;
pub const MLX_ACTION_TOKEN_TYPE_COUNT: usize = 10;
pub const MLX_ACTION_OPERATION_COUNT: usize = 6;

const NONE_CATEGORY: u8 = 5;
const NONE_ARCHETYPE: u8 = u8::MAX;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum MlxActionTokenType {
    ActionMeta = 1,
    LocalPatch = 2,
    BoardObject = 3,
    FrontierObject = 4,
    FrontierTouch = 5,
    ComponentObject = 6,
    MotifObject = 7,
    ComponentKey = 8,
}

impl MlxActionTokenType {
    fn from_code(code: u8) -> Result<Self> {
        match code {
            1 => Ok(Self::ActionMeta),
            2 => Ok(Self::LocalPatch),
            3 => Ok(Self::BoardObject),
            4 => Ok(Self::FrontierObject),
            5 => Ok(Self::FrontierTouch),
            6 => Ok(Self::ComponentObject),
            7 => Ok(Self::MotifObject),
            8 => Ok(Self::ComponentKey),
            _ => Err(R3Error::Invariant(format!(
                "unknown R3 MLX action token type {code}"
            ))),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum MlxActionOperation {
    Context = 0,
    Add = 1,
    Remove = 2,
    UpdateBefore = 3,
    UpdateAfter = 4,
    ControlAfterstate = 5,
}

impl MlxActionOperation {
    fn from_code(code: u8) -> Result<Self> {
        match code {
            0 => Ok(Self::Context),
            1 => Ok(Self::Add),
            2 => Ok(Self::Remove),
            3 => Ok(Self::UpdateBefore),
            4 => Ok(Self::UpdateAfter),
            5 => Ok(Self::ControlAfterstate),
            _ => Err(R3Error::Invariant(format!(
                "unknown R3 MLX action operation {code}"
            ))),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MlxActionToken {
    pub token_type: u8,
    pub operation: u8,
    pub payload: [i8; MLX_ACTION_TOKEN_PAYLOAD_WIDTH],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MlxActionEncoding {
    pub schema_version: u16,
    pub tokens: Vec<MlxActionToken>,
}

impl ActionEdit {
    pub fn mlx_action_encoding(&self) -> Result<MlxActionEncoding> {
        MlxActionEncoding::from_canonical_view(&self.canonical)
    }
}

impl MlxActionEncoding {
    pub fn from_canonical_view(view: &CanonicalActionView) -> Result<Self> {
        let key_catalog = component_key_catalog(&view.global_edit)?;
        let mut tokens = Vec::new();
        tokens.push(meta_token(view)?);
        for cell in &view.local_patch.cells {
            tokens.push(patch_token(cell)?);
        }
        for (index, key) in key_catalog.keys.iter().enumerate() {
            tokens.push(component_key_token(index, key)?);
        }
        encode_global_edit(&mut tokens, &view.global_edit, &key_catalog)?;
        let encoding = Self {
            schema_version: MLX_ACTION_ENCODING_SCHEMA_VERSION,
            tokens,
        };
        if encoding.decode_canonical_view()? != *view {
            return Err(R3Error::Invariant(
                "R3 MLX action token round trip changed the canonical view".to_owned(),
            ));
        }
        Ok(encoding)
    }

    pub fn decode_canonical_view(&self) -> Result<CanonicalActionView> {
        if self.schema_version != MLX_ACTION_ENCODING_SCHEMA_VERSION {
            return Err(R3Error::Invariant(format!(
                "unsupported R3 MLX action encoding schema {}",
                self.schema_version
            )));
        }
        let key_catalog = decode_component_key_catalog(&self.tokens)?;
        let mut meta = None;
        let mut patch = Vec::new();
        let mut board = ObjectCollector::default();
        let mut frontier = FrontierCollector::default();
        let mut components = ObjectCollector::default();
        let mut motifs = ObjectCollector::default();

        for token in &self.tokens {
            let token_type = MlxActionTokenType::from_code(token.token_type)?;
            let operation = MlxActionOperation::from_code(token.operation)?;
            match token_type {
                MlxActionTokenType::ActionMeta => {
                    require_operation(operation, MlxActionOperation::Context)?;
                    if meta.replace(decode_meta(&token.payload)?).is_some() {
                        return Err(R3Error::Invariant(
                            "R3 MLX action encoding has multiple meta tokens".to_owned(),
                        ));
                    }
                }
                MlxActionTokenType::LocalPatch => {
                    require_operation(operation, MlxActionOperation::Context)?;
                    patch.push(decode_patch(&token.payload)?);
                }
                MlxActionTokenType::BoardObject => {
                    let (id, object) = decode_board_object(&token.payload)?;
                    board.insert(operation, id, object)?;
                }
                MlxActionTokenType::FrontierObject => {
                    let (id, object, expected_touches) = decode_frontier_object(&token.payload)?;
                    frontier.insert_object(operation, id, object, expected_touches)?;
                }
                MlxActionTokenType::FrontierTouch => {
                    let decoded = decode_frontier_touch(&token.payload, &key_catalog)?;
                    frontier.insert_touch(operation, decoded)?;
                }
                MlxActionTokenType::ComponentObject => {
                    let (id, object) = decode_component_object(&token.payload)?;
                    components.insert(operation, id, object)?;
                }
                MlxActionTokenType::MotifObject => {
                    let (id, object) = decode_motif_object(&token.payload)?;
                    motifs.insert(operation, id, object)?;
                }
                MlxActionTokenType::ComponentKey => {
                    require_operation(operation, MlxActionOperation::Context)?;
                }
            }
        }

        let (radius, selected_tile, wildlife_destination_offset) = meta.ok_or_else(|| {
            R3Error::Invariant("R3 MLX action encoding is missing its meta token".to_owned())
        })?;
        if radius != 3 {
            return Err(R3Error::Invariant(format!(
                "R3 MLX action encoding has unsupported patch radius {radius}"
            )));
        }
        patch.sort_unstable_by_key(|cell| (cell.offset.q, cell.offset.r));
        if patch.len() != 37
            || patch
                .windows(2)
                .any(|pair| pair[0].offset >= pair[1].offset)
            || patch
                .iter()
                .any(|cell| cell.offset.distance() > u16::from(radius))
        {
            return Err(R3Error::Invariant(
                "R3 MLX local-patch cells are incomplete or noncanonical".to_owned(),
            ));
        }

        let global_edit = CanonicalGlobalEdit {
            board_added: board.take(MlxActionOperation::Add)?,
            board_removed: board.take(MlxActionOperation::Remove)?,
            board_updated: board.take_updates()?,
            frontier_added: frontier.take(MlxActionOperation::Add)?,
            frontier_removed: frontier.take(MlxActionOperation::Remove)?,
            frontier_updated: frontier.take_updates()?,
            components_added: components.take(MlxActionOperation::Add)?,
            components_removed: components.take(MlxActionOperation::Remove)?,
            components_updated: components.take_updates()?,
            motifs_added: motifs.take(MlxActionOperation::Add)?,
            motifs_removed: motifs.take(MlxActionOperation::Remove)?,
            motifs_updated: motifs.take_updates()?,
        };
        board.finish()?;
        frontier.finish()?;
        components.finish()?;
        motifs.finish()?;
        frontier.require_exact_key_usage(&key_catalog)?;

        Ok(CanonicalActionView {
            local_patch: CanonicalLocalPatch {
                radius,
                cells: patch,
            },
            selected_tile,
            wildlife_destination_offset,
            global_edit,
        })
    }
}

fn component_key_catalog(edit: &CanonicalGlobalEdit) -> Result<ComponentKeyCatalog> {
    let mut keys = BTreeSet::new();
    for frontier in edit
        .frontier_added
        .iter()
        .chain(&edit.frontier_removed)
        .chain(
            edit.frontier_updated
                .iter()
                .flat_map(|update| [&update.before, &update.after]),
        )
    {
        for touch in &frontier.touched_habitat_components {
            keys.insert(touch.component_key);
        }
    }
    if keys.len() > usize::from(u8::MAX) + 1 {
        return Err(R3Error::Invariant(
            "R3 MLX component-key catalog exceeds 256 entries".to_owned(),
        ));
    }
    Ok(ComponentKeyCatalog {
        keys: keys.into_iter().collect(),
    })
}

#[derive(Debug, Clone)]
struct ComponentKeyCatalog {
    keys: Vec<[u8; 32]>,
}

impl ComponentKeyCatalog {
    fn index(&self, key: &[u8; 32]) -> Result<u8> {
        let index = self.keys.binary_search(key).map_err(|_| {
            R3Error::Invariant("frontier touch references an absent component key".to_owned())
        })?;
        u8::try_from(index).map_err(Into::into)
    }
}

fn decode_component_key_catalog(tokens: &[MlxActionToken]) -> Result<ComponentKeyCatalog> {
    let mut by_index = BTreeMap::new();
    for token in tokens {
        if MlxActionTokenType::from_code(token.token_type)? != MlxActionTokenType::ComponentKey {
            continue;
        }
        require_operation(
            MlxActionOperation::from_code(token.operation)?,
            MlxActionOperation::Context,
        )?;
        let mut reader = PayloadReader::new(&token.payload);
        let index = reader.raw_u8();
        let mut key = [0u8; 32];
        for value in &mut key {
            *value = reader.raw_u8();
        }
        reader.finish()?;
        if by_index.insert(index, key).is_some() {
            return Err(R3Error::Invariant(
                "R3 MLX component-key catalog repeats an index".to_owned(),
            ));
        }
    }
    for expected in 0..by_index.len() {
        if !by_index.contains_key(&u8::try_from(expected)?) {
            return Err(R3Error::Invariant(
                "R3 MLX component-key catalog indices are not contiguous".to_owned(),
            ));
        }
    }
    Ok(ComponentKeyCatalog {
        keys: by_index.into_values().collect(),
    })
}

fn meta_token(view: &CanonicalActionView) -> Result<MlxActionToken> {
    let mut payload = PayloadWriter::default();
    payload.raw_u8(view.local_patch.radius);
    write_board(&mut payload, &view.selected_tile)?;
    payload.boolean(view.wildlife_destination_offset.is_some());
    if let Some(offset) = view.wildlife_destination_offset {
        payload.coord(offset)?;
    } else {
        payload.signed_i16(0)?;
        payload.signed_i16(0)?;
    }
    token(
        MlxActionTokenType::ActionMeta,
        MlxActionOperation::Context,
        payload,
    )
}

fn decode_meta(
    payload: &[i8; MLX_ACTION_TOKEN_PAYLOAD_WIDTH],
) -> Result<(u8, CanonicalBoardToken, Option<AxialCoord>)> {
    let mut reader = PayloadReader::new(payload);
    let radius = reader.raw_u8();
    let selected = read_board(&mut reader)?;
    let present = reader.boolean()?;
    let destination = reader.coord()?;
    reader.finish()?;
    Ok((radius, selected, present.then_some(destination)))
}

fn patch_token(cell: &LocalPatchCell) -> Result<MlxActionToken> {
    let mut payload = PayloadWriter::default();
    payload.coord(cell.offset)?;
    payload.boolean(cell.inside_rules_grid);
    payload.boolean(cell.frontier);
    payload.boolean(cell.occupied.is_some());
    if let Some(occupied) = &cell.occupied {
        write_board(&mut payload, occupied)?;
    }
    token(
        MlxActionTokenType::LocalPatch,
        MlxActionOperation::Context,
        payload,
    )
}

fn decode_patch(payload: &[i8; MLX_ACTION_TOKEN_PAYLOAD_WIDTH]) -> Result<LocalPatchCell> {
    let mut reader = PayloadReader::new(payload);
    let offset = reader.coord()?;
    let inside_rules_grid = reader.boolean()?;
    let frontier = reader.boolean()?;
    let occupied = if reader.boolean()? {
        let occupied = read_board(&mut reader)?;
        if occupied.offset != offset {
            return Err(R3Error::Invariant(
                "R3 MLX patch cell and occupied token offsets disagree".to_owned(),
            ));
        }
        Some(occupied)
    } else {
        None
    };
    reader.finish()?;
    Ok(LocalPatchCell {
        offset,
        inside_rules_grid,
        frontier,
        occupied,
    })
}

fn component_key_token(index: usize, key: &[u8; 32]) -> Result<MlxActionToken> {
    let mut payload = PayloadWriter::default();
    payload.raw_u8(u8::try_from(index)?);
    for value in key {
        payload.raw_u8(*value);
    }
    token(
        MlxActionTokenType::ComponentKey,
        MlxActionOperation::Context,
        payload,
    )
}

fn encode_global_edit(
    tokens: &mut Vec<MlxActionToken>,
    edit: &CanonicalGlobalEdit,
    key_catalog: &ComponentKeyCatalog,
) -> Result<()> {
    encode_board_list(tokens, &edit.board_added, MlxActionOperation::Add)?;
    encode_board_list(tokens, &edit.board_removed, MlxActionOperation::Remove)?;
    encode_board_updates(tokens, &edit.board_updated)?;
    encode_frontier_list(
        tokens,
        &edit.frontier_added,
        MlxActionOperation::Add,
        key_catalog,
    )?;
    encode_frontier_list(
        tokens,
        &edit.frontier_removed,
        MlxActionOperation::Remove,
        key_catalog,
    )?;
    encode_frontier_updates(tokens, &edit.frontier_updated, key_catalog)?;
    encode_component_list(tokens, &edit.components_added, MlxActionOperation::Add)?;
    encode_component_list(tokens, &edit.components_removed, MlxActionOperation::Remove)?;
    encode_component_updates(tokens, &edit.components_updated)?;
    encode_motif_list(tokens, &edit.motifs_added, MlxActionOperation::Add)?;
    encode_motif_list(tokens, &edit.motifs_removed, MlxActionOperation::Remove)?;
    encode_motif_updates(tokens, &edit.motifs_updated)?;
    Ok(())
}

fn encode_board_list(
    tokens: &mut Vec<MlxActionToken>,
    values: &[CanonicalBoardToken],
    operation: MlxActionOperation,
) -> Result<()> {
    ensure_object_capacity(values.len(), "board")?;
    for (index, value) in values.iter().enumerate() {
        tokens.push(board_object_token(index, operation, value)?);
    }
    Ok(())
}

fn encode_board_updates(
    tokens: &mut Vec<MlxActionToken>,
    values: &[ObjectUpdate<CanonicalBoardToken>],
) -> Result<()> {
    ensure_object_capacity(values.len(), "board update")?;
    for (index, value) in values.iter().enumerate() {
        tokens.push(board_object_token(
            index,
            MlxActionOperation::UpdateBefore,
            &value.before,
        )?);
        tokens.push(board_object_token(
            index,
            MlxActionOperation::UpdateAfter,
            &value.after,
        )?);
    }
    Ok(())
}

fn board_object_token(
    index: usize,
    operation: MlxActionOperation,
    value: &CanonicalBoardToken,
) -> Result<MlxActionToken> {
    let mut payload = PayloadWriter::default();
    payload.raw_u8(u8::try_from(index)?);
    write_board(&mut payload, value)?;
    token(MlxActionTokenType::BoardObject, operation, payload)
}

fn decode_board_object(
    payload: &[i8; MLX_ACTION_TOKEN_PAYLOAD_WIDTH],
) -> Result<(u8, CanonicalBoardToken)> {
    let mut reader = PayloadReader::new(payload);
    let id = reader.raw_u8();
    let object = read_board(&mut reader)?;
    reader.finish()?;
    Ok((id, object))
}

fn encode_frontier_list(
    tokens: &mut Vec<MlxActionToken>,
    values: &[CanonicalFrontierToken],
    operation: MlxActionOperation,
    key_catalog: &ComponentKeyCatalog,
) -> Result<()> {
    ensure_object_capacity(values.len(), "frontier")?;
    for (index, value) in values.iter().enumerate() {
        encode_frontier(tokens, index, operation, value, key_catalog)?;
    }
    Ok(())
}

fn encode_frontier_updates(
    tokens: &mut Vec<MlxActionToken>,
    values: &[ObjectUpdate<CanonicalFrontierToken>],
    key_catalog: &ComponentKeyCatalog,
) -> Result<()> {
    ensure_object_capacity(values.len(), "frontier update")?;
    for (index, value) in values.iter().enumerate() {
        encode_frontier(
            tokens,
            index,
            MlxActionOperation::UpdateBefore,
            &value.before,
            key_catalog,
        )?;
        encode_frontier(
            tokens,
            index,
            MlxActionOperation::UpdateAfter,
            &value.after,
            key_catalog,
        )?;
    }
    Ok(())
}

fn encode_frontier(
    tokens: &mut Vec<MlxActionToken>,
    index: usize,
    operation: MlxActionOperation,
    value: &CanonicalFrontierToken,
    key_catalog: &ComponentKeyCatalog,
) -> Result<()> {
    if value.touched_habitat_components.len() > 6 {
        return Err(R3Error::Invariant(
            "canonical frontier touches more than six components".to_owned(),
        ));
    }
    let mut payload = PayloadWriter::default();
    payload.raw_u8(u8::try_from(index)?);
    payload.coord(value.offset)?;
    payload.raw_u8(value.neighbor_presence_bits);
    for terrain in value.neighbor_facing_terrains {
        payload.raw_u8(optional_terrain_code(terrain));
    }
    for count in value.adjacent_wildlife_counts {
        payload.raw_u8(count);
    }
    payload.raw_u8(value.occupied_neighbor_runs);
    payload.raw_u8(value.opposite_neighbor_pair_bits);
    payload.raw_u8(u8::try_from(value.touched_habitat_components.len())?);
    for size in value.resulting_size_by_terrain {
        payload.small_u16(size)?;
    }
    payload.raw_u8(value.habitat_bridge_terrain_bits);
    payload.raw_u8(value.repeated_component_contact_terrain_bits);
    tokens.push(token(
        MlxActionTokenType::FrontierObject,
        operation,
        payload,
    )?);
    for (ordinal, touch) in value.touched_habitat_components.iter().enumerate() {
        let mut payload = PayloadWriter::default();
        payload.raw_u8(u8::try_from(index)?);
        payload.raw_u8(u8::try_from(ordinal)?);
        payload.raw_u8(touch.terrain as u8);
        payload.raw_u8(key_catalog.index(&touch.component_key)?);
        payload.small_u16(touch.component_size)?;
        payload.raw_u8(touch.contact_edge_bits);
        tokens.push(token(
            MlxActionTokenType::FrontierTouch,
            operation,
            payload,
        )?);
    }
    Ok(())
}

fn decode_frontier_object(
    payload: &[i8; MLX_ACTION_TOKEN_PAYLOAD_WIDTH],
) -> Result<(u8, CanonicalFrontierToken, usize)> {
    let mut reader = PayloadReader::new(payload);
    let id = reader.raw_u8();
    let offset = reader.coord()?;
    let neighbor_presence_bits = reader.raw_u8();
    let mut neighbor_facing_terrains = [None; 6];
    for value in &mut neighbor_facing_terrains {
        *value = optional_terrain(reader.raw_u8())?;
    }
    let mut adjacent_wildlife_counts = [0u8; 5];
    for value in &mut adjacent_wildlife_counts {
        *value = reader.raw_u8();
    }
    let occupied_neighbor_runs = reader.raw_u8();
    let opposite_neighbor_pair_bits = reader.raw_u8();
    let expected_touches = usize::from(reader.raw_u8());
    let mut resulting_size_by_terrain = [0u16; 5];
    for value in &mut resulting_size_by_terrain {
        *value = reader.small_u16()?;
    }
    let habitat_bridge_terrain_bits = reader.raw_u8();
    let repeated_component_contact_terrain_bits = reader.raw_u8();
    reader.finish()?;
    Ok((
        id,
        CanonicalFrontierToken {
            offset,
            neighbor_presence_bits,
            neighbor_facing_terrains,
            adjacent_wildlife_counts,
            occupied_neighbor_runs,
            opposite_neighbor_pair_bits,
            touched_habitat_components: Vec::new(),
            resulting_size_by_terrain,
            habitat_bridge_terrain_bits,
            repeated_component_contact_terrain_bits,
        },
        expected_touches,
    ))
}

#[derive(Debug)]
struct DecodedFrontierTouch {
    parent_id: u8,
    ordinal: u8,
    touch: CanonicalFrontierTouch,
}

fn decode_frontier_touch(
    payload: &[i8; MLX_ACTION_TOKEN_PAYLOAD_WIDTH],
    key_catalog: &ComponentKeyCatalog,
) -> Result<DecodedFrontierTouch> {
    let mut reader = PayloadReader::new(payload);
    let parent_id = reader.raw_u8();
    let ordinal = reader.raw_u8();
    let terrain = terrain(reader.raw_u8())?;
    let key_index = usize::from(reader.raw_u8());
    let component_key = *key_catalog.keys.get(key_index).ok_or_else(|| {
        R3Error::Invariant("frontier touch component-key index is absent".to_owned())
    })?;
    let component_size = reader.small_u16()?;
    let contact_edge_bits = reader.raw_u8();
    reader.finish()?;
    Ok(DecodedFrontierTouch {
        parent_id,
        ordinal,
        touch: CanonicalFrontierTouch {
            terrain,
            component_key,
            component_size,
            contact_edge_bits,
        },
    })
}

fn encode_component_list(
    tokens: &mut Vec<MlxActionToken>,
    values: &[CanonicalComponentObject],
    operation: MlxActionOperation,
) -> Result<()> {
    ensure_object_capacity(values.len(), "component")?;
    for (index, value) in values.iter().enumerate() {
        tokens.push(component_object_token(index, operation, value)?);
    }
    Ok(())
}

fn encode_component_updates(
    tokens: &mut Vec<MlxActionToken>,
    values: &[ObjectUpdate<CanonicalComponentObject>],
) -> Result<()> {
    ensure_object_capacity(values.len(), "component update")?;
    for (index, value) in values.iter().enumerate() {
        tokens.push(component_object_token(
            index,
            MlxActionOperation::UpdateBefore,
            &value.before,
        )?);
        tokens.push(component_object_token(
            index,
            MlxActionOperation::UpdateAfter,
            &value.after,
        )?);
    }
    Ok(())
}

fn component_object_token(
    index: usize,
    operation: MlxActionOperation,
    value: &CanonicalComponentObject,
) -> Result<MlxActionToken> {
    if value.members.len() != usize::from(value.member_count) || value.members.len() > 23 {
        return Err(R3Error::Invariant(
            "canonical component member accounting is invalid".to_owned(),
        ));
    }
    let mut payload = PayloadWriter::default();
    payload.raw_u8(u8::try_from(index)?);
    payload.raw_u8(value.terrain as u8);
    payload.raw_u8(u8::try_from(value.members.len())?);
    payload.small_u16(value.matching_internal_edge_count)?;
    payload.small_u16(value.open_boundary_edge_count)?;
    payload.small_u16(value.frontier_contact_count)?;
    for member in &value.members {
        payload.coord(*member)?;
    }
    token(MlxActionTokenType::ComponentObject, operation, payload)
}

fn decode_component_object(
    payload: &[i8; MLX_ACTION_TOKEN_PAYLOAD_WIDTH],
) -> Result<(u8, CanonicalComponentObject)> {
    let mut reader = PayloadReader::new(payload);
    let id = reader.raw_u8();
    let terrain = terrain(reader.raw_u8())?;
    let member_count = usize::from(reader.raw_u8());
    if member_count > 23 {
        return Err(R3Error::Invariant(
            "R3 MLX component exceeds the 23-member board bound".to_owned(),
        ));
    }
    let matching_internal_edge_count = reader.small_u16()?;
    let open_boundary_edge_count = reader.small_u16()?;
    let frontier_contact_count = reader.small_u16()?;
    let mut members = Vec::with_capacity(member_count);
    for _ in 0..member_count {
        members.push(reader.coord()?);
    }
    reader.finish()?;
    Ok((
        id,
        CanonicalComponentObject {
            terrain,
            member_count: u16::try_from(member_count)?,
            members,
            matching_internal_edge_count,
            open_boundary_edge_count,
            frontier_contact_count,
        },
    ))
}

fn encode_motif_list(
    tokens: &mut Vec<MlxActionToken>,
    values: &[CanonicalMotifObject],
    operation: MlxActionOperation,
) -> Result<()> {
    ensure_object_capacity(values.len(), "motif")?;
    for (index, value) in values.iter().enumerate() {
        tokens.push(motif_object_token(index, operation, value)?);
    }
    Ok(())
}

fn encode_motif_updates(
    tokens: &mut Vec<MlxActionToken>,
    values: &[ObjectUpdate<CanonicalMotifObject>],
) -> Result<()> {
    ensure_object_capacity(values.len(), "motif update")?;
    for (index, value) in values.iter().enumerate() {
        tokens.push(motif_object_token(
            index,
            MlxActionOperation::UpdateBefore,
            &value.before,
        )?);
        tokens.push(motif_object_token(
            index,
            MlxActionOperation::UpdateAfter,
            &value.after,
        )?);
    }
    Ok(())
}

fn motif_object_token(
    index: usize,
    operation: MlxActionOperation,
    value: &CanonicalMotifObject,
) -> Result<MlxActionToken> {
    let mut payload = PayloadWriter::default();
    payload.raw_u8(u8::try_from(index)?);
    payload.coord(value.offset)?;
    payload.raw_u8(value.wildlife as u8);
    for wildlife in value.neighbor_wildlife {
        payload.raw_u8(optional_wildlife_code(wildlife));
    }
    for count in value.adjacent_wildlife_counts {
        payload.raw_u8(count);
    }
    payload.raw_u8(value.same_species_neighbor_bits);
    token(MlxActionTokenType::MotifObject, operation, payload)
}

fn decode_motif_object(
    payload: &[i8; MLX_ACTION_TOKEN_PAYLOAD_WIDTH],
) -> Result<(u8, CanonicalMotifObject)> {
    let mut reader = PayloadReader::new(payload);
    let id = reader.raw_u8();
    let offset = reader.coord()?;
    let wildlife = wildlife(reader.raw_u8())?;
    let mut neighbor_wildlife = [None; 6];
    for value in &mut neighbor_wildlife {
        *value = optional_wildlife(reader.raw_u8())?;
    }
    let mut adjacent_wildlife_counts = [0u8; 5];
    for value in &mut adjacent_wildlife_counts {
        *value = reader.raw_u8();
    }
    let same_species_neighbor_bits = reader.raw_u8();
    reader.finish()?;
    Ok((
        id,
        CanonicalMotifObject {
            offset,
            wildlife,
            neighbor_wildlife,
            adjacent_wildlife_counts,
            same_species_neighbor_bits,
        },
    ))
}

fn write_board(payload: &mut PayloadWriter, value: &CanonicalBoardToken) -> Result<()> {
    payload.coord(value.offset)?;
    payload.raw_u8(value.tile.terrain_a as u8);
    payload.raw_u8(optional_terrain_code(value.tile.terrain_b));
    payload.raw_u8(value.tile.wildlife_eligibility.bits());
    payload.boolean(value.tile.keystone);
    payload.raw_u8(
        value
            .tile
            .semantic_archetype_id
            .map(u8::try_from)
            .transpose()?
            .unwrap_or(NONE_ARCHETYPE),
    );
    if value.rotation >= 6 {
        return Err(R3Error::Invariant(
            "canonical board token rotation is outside [0, 5]".to_owned(),
        ));
    }
    payload.raw_u8(value.rotation);
    for terrain in value.directed_edge_terrains {
        payload.raw_u8(terrain as u8);
    }
    payload.raw_u8(optional_wildlife_code(value.placed_wildlife));
    Ok(())
}

fn read_board(payload: &mut PayloadReader<'_>) -> Result<CanonicalBoardToken> {
    let offset = payload.coord()?;
    let terrain_a = terrain(payload.raw_u8())?;
    let terrain_b = optional_terrain(payload.raw_u8())?;
    let wildlife_bits = payload.raw_u8();
    let wildlife_eligibility = WildlifeMask::from_bits(wildlife_bits);
    if wildlife_eligibility.bits() != wildlife_bits {
        return Err(R3Error::Invariant(
            "R3 MLX board token wildlife mask is invalid".to_owned(),
        ));
    }
    let keystone = payload.boolean()?;
    let semantic_archetype_id = match payload.raw_u8() {
        NONE_ARCHETYPE => None,
        value => Some(u16::from(value)),
    };
    let rotation = payload.raw_u8();
    if rotation >= 6 {
        return Err(R3Error::Invariant(
            "R3 MLX board token rotation is outside [0, 5]".to_owned(),
        ));
    }
    let mut directed_edge_terrains = [Terrain::Mountain; 6];
    for value in &mut directed_edge_terrains {
        *value = terrain(payload.raw_u8())?;
    }
    let placed_wildlife = optional_wildlife(payload.raw_u8())?;
    let tile = TileSemantic::new(terrain_a, terrain_b, wildlife_eligibility, keystone);
    if tile.semantic_archetype_id != semantic_archetype_id {
        return Err(R3Error::Invariant(
            "R3 MLX board token semantic archetype is inconsistent".to_owned(),
        ));
    }
    Ok(CanonicalBoardToken {
        offset,
        tile,
        rotation,
        directed_edge_terrains,
        placed_wildlife,
    })
}

fn token(
    token_type: MlxActionTokenType,
    operation: MlxActionOperation,
    payload: PayloadWriter,
) -> Result<MlxActionToken> {
    Ok(MlxActionToken {
        token_type: token_type as u8,
        operation: operation as u8,
        payload: payload.finish()?,
    })
}

fn ensure_object_capacity(count: usize, label: &str) -> Result<()> {
    if count > usize::from(u8::MAX) + 1 {
        return Err(R3Error::Invariant(format!(
            "R3 MLX {label} collection exceeds 256 objects"
        )));
    }
    Ok(())
}

fn require_operation(actual: MlxActionOperation, expected: MlxActionOperation) -> Result<()> {
    if actual != expected {
        return Err(R3Error::Invariant(format!(
            "R3 MLX token operation {:?} is invalid for this token; expected {:?}",
            actual, expected
        )));
    }
    Ok(())
}

struct ObjectCollector<T> {
    objects: BTreeMap<(MlxActionOperation, u8), T>,
}

impl<T> Default for ObjectCollector<T> {
    fn default() -> Self {
        Self {
            objects: BTreeMap::new(),
        }
    }
}

impl<T> ObjectCollector<T> {
    fn insert(&mut self, operation: MlxActionOperation, id: u8, value: T) -> Result<()> {
        if operation == MlxActionOperation::Context
            || operation == MlxActionOperation::ControlAfterstate
        {
            return Err(R3Error::Invariant(
                "R3 MLX global object uses a non-edit operation".to_owned(),
            ));
        }
        if self.objects.insert((operation, id), value).is_some() {
            return Err(R3Error::Invariant(
                "R3 MLX global object repeats an operation/id pair".to_owned(),
            ));
        }
        Ok(())
    }

    fn take(&mut self, operation: MlxActionOperation) -> Result<Vec<T>> {
        ordered_take(&mut self.objects, operation)
    }

    fn take_updates(&mut self) -> Result<Vec<ObjectUpdate<T>>> {
        let before = self.take(MlxActionOperation::UpdateBefore)?;
        let after = self.take(MlxActionOperation::UpdateAfter)?;
        if before.len() != after.len() {
            return Err(R3Error::Invariant(
                "R3 MLX update before/after lengths disagree".to_owned(),
            ));
        }
        Ok(before
            .into_iter()
            .zip(after)
            .map(|(before, after)| ObjectUpdate { before, after })
            .collect())
    }

    fn finish(&self) -> Result<()> {
        if self.objects.is_empty() {
            Ok(())
        } else {
            Err(R3Error::Invariant(
                "R3 MLX object collector retained unsupported operations".to_owned(),
            ))
        }
    }
}

fn ordered_take<T>(
    values: &mut BTreeMap<(MlxActionOperation, u8), T>,
    operation: MlxActionOperation,
) -> Result<Vec<T>> {
    let keys = values
        .keys()
        .filter(|(candidate, _)| *candidate == operation)
        .copied()
        .collect::<Vec<_>>();
    for (expected, (_, id)) in keys.iter().enumerate() {
        if usize::from(*id) != expected {
            return Err(R3Error::Invariant(
                "R3 MLX object IDs are not contiguous".to_owned(),
            ));
        }
    }
    Ok(keys
        .into_iter()
        .map(|key| {
            values
                .remove(&key)
                .expect("key was collected from this map")
        })
        .collect())
}

struct FrontierCollector {
    objects: ObjectCollector<CanonicalFrontierToken>,
    expected_touches: BTreeMap<(MlxActionOperation, u8), usize>,
    touches: BTreeMap<(MlxActionOperation, u8, u8), CanonicalFrontierTouch>,
    used_keys: BTreeSet<[u8; 32]>,
}

impl Default for FrontierCollector {
    fn default() -> Self {
        Self {
            objects: ObjectCollector::default(),
            expected_touches: BTreeMap::new(),
            touches: BTreeMap::new(),
            used_keys: BTreeSet::new(),
        }
    }
}

impl FrontierCollector {
    fn insert_object(
        &mut self,
        operation: MlxActionOperation,
        id: u8,
        value: CanonicalFrontierToken,
        expected_touches: usize,
    ) -> Result<()> {
        if expected_touches > 6
            || self
                .expected_touches
                .insert((operation, id), expected_touches)
                .is_some()
        {
            return Err(R3Error::Invariant(
                "R3 MLX frontier touch count is invalid or duplicated".to_owned(),
            ));
        }
        self.objects.insert(operation, id, value)
    }

    fn insert_touch(
        &mut self,
        operation: MlxActionOperation,
        value: DecodedFrontierTouch,
    ) -> Result<()> {
        if operation == MlxActionOperation::Context
            || operation == MlxActionOperation::ControlAfterstate
            || self
                .touches
                .insert(
                    (operation, value.parent_id, value.ordinal),
                    value.touch.clone(),
                )
                .is_some()
        {
            return Err(R3Error::Invariant(
                "R3 MLX frontier touch operation or identity is invalid".to_owned(),
            ));
        }
        self.used_keys.insert(value.touch.component_key);
        Ok(())
    }

    fn attach_touches(&mut self) -> Result<()> {
        let keys = self.expected_touches.keys().copied().collect::<Vec<_>>();
        for (operation, id) in keys {
            let expected = self.expected_touches.remove(&(operation, id)).unwrap();
            let object = self
                .objects
                .objects
                .get_mut(&(operation, id))
                .ok_or_else(|| {
                    R3Error::Invariant(
                        "R3 MLX frontier touch references an absent object".to_owned(),
                    )
                })?;
            for ordinal in 0..expected {
                object.touched_habitat_components.push(
                    self.touches
                        .remove(&(operation, id, u8::try_from(ordinal)?))
                        .ok_or_else(|| {
                            R3Error::Invariant(
                                "R3 MLX frontier touch ordinals are incomplete".to_owned(),
                            )
                        })?,
                );
            }
        }
        if !self.touches.is_empty() {
            return Err(R3Error::Invariant(
                "R3 MLX frontier has unclaimed touch tokens".to_owned(),
            ));
        }
        Ok(())
    }

    fn take(&mut self, operation: MlxActionOperation) -> Result<Vec<CanonicalFrontierToken>> {
        self.attach_touches()?;
        self.objects.take(operation)
    }

    fn take_updates(&mut self) -> Result<Vec<ObjectUpdate<CanonicalFrontierToken>>> {
        self.attach_touches()?;
        self.objects.take_updates()
    }

    fn finish(&self) -> Result<()> {
        if self.expected_touches.is_empty() && self.touches.is_empty() {
            self.objects.finish()
        } else {
            Err(R3Error::Invariant(
                "R3 MLX frontier collector retained incomplete touches".to_owned(),
            ))
        }
    }

    fn require_exact_key_usage(&self, catalog: &ComponentKeyCatalog) -> Result<()> {
        let expected = catalog.keys.iter().copied().collect::<BTreeSet<_>>();
        if self.used_keys != expected {
            return Err(R3Error::Invariant(
                "R3 MLX component-key catalog contains missing or unused entries".to_owned(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct PayloadWriter {
    bytes: [i8; MLX_ACTION_TOKEN_PAYLOAD_WIDTH],
    cursor: usize,
    overflowed: bool,
}

impl Default for PayloadWriter {
    fn default() -> Self {
        Self {
            bytes: [0; MLX_ACTION_TOKEN_PAYLOAD_WIDTH],
            cursor: 0,
            overflowed: false,
        }
    }
}

impl PayloadWriter {
    fn raw_u8(&mut self, value: u8) {
        self.write(value as i8);
    }

    fn boolean(&mut self, value: bool) {
        self.raw_u8(u8::from(value));
    }

    fn small_u16(&mut self, value: u16) -> Result<()> {
        self.write(i8::try_from(value).map_err(|_| {
            R3Error::Invariant(format!(
                "R3 MLX payload value {value} exceeds signed-byte range"
            ))
        })?);
        Ok(())
    }

    fn signed_i16(&mut self, value: i16) -> Result<()> {
        self.write(i8::try_from(value).map_err(|_| {
            R3Error::Invariant(format!(
                "R3 MLX coordinate/value {value} exceeds signed-byte range"
            ))
        })?);
        Ok(())
    }

    fn coord(&mut self, value: AxialCoord) -> Result<()> {
        self.signed_i16(value.q)?;
        self.signed_i16(value.r)
    }

    fn write(&mut self, value: i8) {
        if self.cursor >= MLX_ACTION_TOKEN_PAYLOAD_WIDTH {
            self.overflowed = true;
            return;
        }
        self.bytes[self.cursor] = value;
        self.cursor += 1;
    }

    fn finish(self) -> Result<[i8; MLX_ACTION_TOKEN_PAYLOAD_WIDTH]> {
        if self.overflowed {
            return Err(R3Error::Invariant(format!(
                "R3 MLX token payload exceeds the frozen {}-byte width",
                MLX_ACTION_TOKEN_PAYLOAD_WIDTH
            )));
        }
        Ok(self.bytes)
    }
}

struct PayloadReader<'a> {
    bytes: &'a [i8; MLX_ACTION_TOKEN_PAYLOAD_WIDTH],
    cursor: usize,
}

impl<'a> PayloadReader<'a> {
    fn new(bytes: &'a [i8; MLX_ACTION_TOKEN_PAYLOAD_WIDTH]) -> Self {
        Self { bytes, cursor: 0 }
    }

    fn raw_u8(&mut self) -> u8 {
        let value = self.bytes[self.cursor] as u8;
        self.cursor += 1;
        value
    }

    fn boolean(&mut self) -> Result<bool> {
        match self.raw_u8() {
            0 => Ok(false),
            1 => Ok(true),
            value => Err(R3Error::Invariant(format!(
                "R3 MLX boolean payload is {value}, not zero or one"
            ))),
        }
    }

    fn small_u16(&mut self) -> Result<u16> {
        let value = self.bytes[self.cursor];
        self.cursor += 1;
        if value < 0 {
            return Err(R3Error::Invariant(
                "R3 MLX nonnegative payload value is negative".to_owned(),
            ));
        }
        Ok(value as u16)
    }

    fn signed_i16(&mut self) -> i16 {
        let value = self.bytes[self.cursor];
        self.cursor += 1;
        i16::from(value)
    }

    fn coord(&mut self) -> Result<AxialCoord> {
        Ok(AxialCoord::new(self.signed_i16(), self.signed_i16()))
    }

    fn finish(&self) -> Result<()> {
        if self.bytes[self.cursor..].iter().any(|value| *value != 0) {
            return Err(R3Error::Invariant(
                "R3 MLX payload has nonzero trailing bytes".to_owned(),
            ));
        }
        Ok(())
    }
}

fn terrain(code: u8) -> Result<Terrain> {
    Terrain::ALL
        .get(usize::from(code))
        .copied()
        .ok_or_else(|| R3Error::Invariant(format!("R3 MLX terrain code {code} is invalid")))
}

fn optional_terrain(code: u8) -> Result<Option<Terrain>> {
    if code == NONE_CATEGORY {
        Ok(None)
    } else {
        terrain(code).map(Some)
    }
}

fn optional_terrain_code(value: Option<Terrain>) -> u8 {
    value.map_or(NONE_CATEGORY, |terrain| terrain as u8)
}

fn wildlife(code: u8) -> Result<Wildlife> {
    Wildlife::ALL
        .get(usize::from(code))
        .copied()
        .ok_or_else(|| R3Error::Invariant(format!("R3 MLX wildlife code {code} is invalid")))
}

fn optional_wildlife(code: u8) -> Result<Option<Wildlife>> {
    if code == NONE_CATEGORY {
        Ok(None)
    } else {
        wildlife(code).map(Some)
    }
}

fn optional_wildlife_code(value: Option<Wildlife>) -> u8 {
    value.map_or(NONE_CATEGORY, |wildlife| wildlife as u8)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn payload_overflow_fails_closed_without_panicking() {
        let mut payload = PayloadWriter::default();
        for _ in 0..=MLX_ACTION_TOKEN_PAYLOAD_WIDTH {
            payload.raw_u8(1);
        }
        let error = token(
            MlxActionTokenType::ActionMeta,
            MlxActionOperation::Context,
            payload,
        )
        .unwrap_err();
        assert!(error.to_string().contains("exceeds the frozen"));
    }
}
