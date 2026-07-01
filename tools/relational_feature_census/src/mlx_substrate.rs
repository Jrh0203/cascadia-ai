use cascadia_game::D6Transform;
use r2_sparse_entity_census::{AxialCoord, SparsePublicState};

use crate::{BoardGraph, RelationalStateGraph, Result, invalid};

pub const RELATIONAL_TOKEN_CLASS_COUNT: usize = 8;
pub const RELATIONAL_TOKEN_VALUE_WIDTH: usize = 64;
pub const R5_MINIMAL_CLASS_COUNT: u8 = 6;

pub const HABITAT_COMPONENT_CLASS: u8 = 1;
pub const BEAR_COMPONENT_CLASS: u8 = 2;
pub const ELK_LINE_CLASS: u8 = 3;
pub const SALMON_COMPONENT_CLASS: u8 = 4;
pub const HAWK_POSITION_CLASS: u8 = 5;
pub const FOX_CENTER_CLASS: u8 = 6;
pub const FRONTIER_SUMMARY_CLASS: u8 = 7;
pub const OPPORTUNITY_SUMMARY_CLASS: u8 = 8;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelationalParentToken {
    pub class_id: u8,
    pub relative_seat: u8,
    pub values: [i16; RELATIONAL_TOKEN_VALUE_WIDTH],
}

pub fn rich_relational_parent_tokens(
    sparse: &SparsePublicState,
    transform: D6Transform,
) -> Result<Vec<RelationalParentToken>> {
    let transformed = sparse.transformed(transform)?;
    let graph = RelationalStateGraph::from_sparse(&transformed)?;
    let mut tokens = Vec::new();
    for board in &graph.boards {
        encode_board(board, &mut tokens)?;
    }
    Ok(tokens)
}

pub fn r5_minimal_token(token: &RelationalParentToken) -> Option<RelationalParentToken> {
    if token.class_id > R5_MINIMAL_CLASS_COUNT {
        return None;
    }
    let mut minimal = token.clone();
    match minimal.class_id {
        HABITAT_COMPONENT_CLASS => minimal.values[42..].fill(0),
        BEAR_COMPONENT_CLASS | ELK_LINE_CLASS => {}
        SALMON_COMPONENT_CLASS => {
            let member_count = usize::try_from(minimal.values[0]).ok()?;
            minimal.values[1..4].fill(0);
            minimal.values[5] = 0;
            let continuation_start = 6usize.checked_add(member_count.checked_mul(2)?)?;
            if continuation_start < RELATIONAL_TOKEN_VALUE_WIDTH {
                minimal.values[continuation_start..].fill(0);
            }
        }
        HAWK_POSITION_CLASS => minimal.values[2] = 0,
        FOX_CENTER_CLASS => minimal.values[3..].fill(0),
        _ => return None,
    }
    Some(minimal)
}

fn encode_board(board: &BoardGraph, tokens: &mut Vec<RelationalParentToken>) -> Result<()> {
    for component in &board.habitat_components {
        let mut values = [0; RELATIONAL_TOKEN_VALUE_WIDTH];
        values[0] = component.terrain as i16;
        values[1] = as_i16(component.member_count)?;
        write_coords(&mut values, 2, &component.members)?;
        values[42] = as_i16(component.matching_internal_edge_count)?;
        values[43] = as_i16(component.open_boundary_edge_count)?;
        values[44] = as_i16(component.frontier_contact_count)?;
        values[45] = as_i16(component.cycle_rank)?;
        values[46] = as_i16(component.bridge_count)?;
        values[47] = as_i16(component.articulation_count)?;
        values[48] = as_i16(component.size_rank)?;
        values[49] = as_i16(component.merge_frontier_count)?;
        values[50] = as_i16(component.largest_merge_result)?;
        push(tokens, HABITAT_COMPONENT_CLASS, board.relative_seat, values)?;
    }
    for component in &board.bear_components {
        let mut values = [0; RELATIONAL_TOKEN_VALUE_WIDTH];
        values[0] = as_i16(component.members.len())?;
        values[1] = as_i16(component.edge_count)?;
        values[2] = as_i16(component.endpoint_count)?;
        values[3] = i16::from(component.maximum_degree);
        write_coords(&mut values, 4, &component.members)?;
        push(tokens, BEAR_COMPONENT_CLASS, board.relative_seat, values)?;
    }
    for line in &board.elk_lines {
        let mut values = [0; RELATIONAL_TOKEN_VALUE_WIDTH];
        values[0] = i16::from(line.axis);
        values[1] = as_i16(line.members.len())?;
        values[2] = i16::from(line.eligible_extension_count);
        write_coord(&mut values, 3, line.negative_extension)?;
        write_coord(&mut values, 5, line.positive_extension)?;
        write_coords(&mut values, 7, &line.members)?;
        push(tokens, ELK_LINE_CLASS, board.relative_seat, values)?;
    }
    for component in &board.salmon_components {
        let mut values = [0; RELATIONAL_TOKEN_VALUE_WIDTH];
        values[0] = as_i16(component.members.len())?;
        values[1] = as_i16(component.edge_count)?;
        values[2] = as_i16(component.endpoint_count)?;
        values[3] = as_i16(component.branch_conflict_count)?;
        values[4] = i16::from(component.valid_run);
        values[5] = as_i16(component.legal_continuations.len())?;
        let next = write_coords(&mut values, 6, &component.members)?;
        write_coords(&mut values, next, &component.legal_continuations)?;
        push(tokens, SALMON_COMPONENT_CLASS, board.relative_seat, values)?;
    }
    for hawk in &board.hawk_positions {
        let mut values = [0; RELATIONAL_TOKEN_VALUE_WIDTH];
        write_coord(&mut values, 0, *hawk)?;
        let conflict_degree = board
            .hawk_conflict_edges
            .iter()
            .filter(|edge| edge[0] == *hawk || edge[1] == *hawk)
            .count();
        values[2] = as_i16(conflict_degree)?;
        values[3] = i16::from(conflict_degree == 0);
        push(tokens, HAWK_POSITION_CLASS, board.relative_seat, values)?;
    }
    for center in &board.fox_centers {
        let mut values = [0; RELATIONAL_TOKEN_VALUE_WIDTH];
        write_coord(&mut values, 0, center.coord)?;
        values[2] = i16::from(center.neighbor_diversity_mask);
        values[3] = i16::from(center.missing_wildlife_mask);
        values[4] = as_i16(center.compatible_cells.len())?;
        write_coords(&mut values, 5, &center.compatible_cells)?;
        push(tokens, FOX_CENTER_CLASS, board.relative_seat, values)?;
    }

    let frontier = &board.frontier;
    let mut values = [0; RELATIONAL_TOKEN_VALUE_WIDTH];
    values[0] = as_i16(frontier.frontier_count)?;
    write_u16_array(&mut values, 1, &frontier.degree_histogram)?;
    write_u16_array(&mut values, 8, &frontier.bridge_frontiers_by_terrain)?;
    write_u16_array(
        &mut values,
        13,
        &frontier.repeated_contact_frontiers_by_terrain,
    )?;
    write_u16_array(&mut values, 18, &frontier.maximum_resulting_size_by_terrain)?;
    write_u32_array(&mut values, 23, &frontier.sum_resulting_size_by_terrain)?;
    push(tokens, FRONTIER_SUMMARY_CLASS, board.relative_seat, values)?;

    let opportunity = &board.opportunity;
    let mut values = [0; RELATIONAL_TOKEN_VALUE_WIDTH];
    write_u16_array(&mut values, 0, &opportunity.eligible_empty_cells)?;
    values[5] = as_i16(opportunity.bear_singletons)?;
    values[6] = as_i16(opportunity.bear_pairs)?;
    values[7] = as_i16(opportunity.bear_oversize_components)?;
    values[8] = as_i16(opportunity.bear_pair_completion_cells)?;
    values[9] = as_i16(opportunity.bear_oversize_risk_cells)?;
    write_u16_array(&mut values, 10, &opportunity.elk_lines_by_length)?;
    values[15] = as_i16(opportunity.elk_eligible_extensions)?;
    values[16] = as_i16(opportunity.elk_overlapping_members)?;
    values[17] = as_i16(opportunity.salmon_valid_runs)?;
    values[18] = as_i16(opportunity.salmon_invalid_components)?;
    values[19] = as_i16(opportunity.salmon_endpoints)?;
    values[20] = as_i16(opportunity.salmon_branch_conflicts)?;
    values[21] = as_i16(opportunity.salmon_legal_continuations)?;
    values[22] = as_i16(opportunity.hawk_conflict_edges)?;
    values[23] = as_i16(opportunity.hawk_isolated)?;
    values[24] = as_i16(opportunity.hawk_isolated_opportunities)?;
    values[25] = as_i16(opportunity.fox_centers)?;
    values[26] = as_i16(opportunity.fox_diversity_sum)?;
    values[27] = as_i16(opportunity.fox_missing_types)?;
    values[28] = as_i16(opportunity.fox_compatible_cells)?;
    push(
        tokens,
        OPPORTUNITY_SUMMARY_CLASS,
        board.relative_seat,
        values,
    )
}

fn push(
    tokens: &mut Vec<RelationalParentToken>,
    class_id: u8,
    relative_seat: u8,
    values: [i16; RELATIONAL_TOKEN_VALUE_WIDTH],
) -> Result<()> {
    if !(1..=RELATIONAL_TOKEN_CLASS_COUNT as u8).contains(&class_id) {
        return Err(invalid("relational parent token class is out of range"));
    }
    tokens.push(RelationalParentToken {
        class_id,
        relative_seat,
        values,
    });
    Ok(())
}

fn write_coord(
    values: &mut [i16; RELATIONAL_TOKEN_VALUE_WIDTH],
    start: usize,
    coord: AxialCoord,
) -> Result<usize> {
    if start + 2 > values.len() {
        return Err(invalid("relational token coordinate payload overflowed"));
    }
    values[start] = coord.q;
    values[start + 1] = coord.r;
    Ok(start + 2)
}

fn write_coords(
    values: &mut [i16; RELATIONAL_TOKEN_VALUE_WIDTH],
    mut start: usize,
    coords: &[AxialCoord],
) -> Result<usize> {
    for coord in coords {
        start = write_coord(values, start, *coord)?;
    }
    Ok(start)
}

fn write_u16_array<const N: usize>(
    values: &mut [i16; RELATIONAL_TOKEN_VALUE_WIDTH],
    start: usize,
    source: &[u16; N],
) -> Result<()> {
    if start + N > values.len() {
        return Err(invalid("relational token u16 array overflowed"));
    }
    for (offset, value) in source.iter().enumerate() {
        values[start + offset] = as_i16(*value)?;
    }
    Ok(())
}

fn write_u32_array<const N: usize>(
    values: &mut [i16; RELATIONAL_TOKEN_VALUE_WIDTH],
    start: usize,
    source: &[u32; N],
) -> Result<()> {
    if start + N > values.len() {
        return Err(invalid("relational token u32 array overflowed"));
    }
    for (offset, value) in source.iter().enumerate() {
        values[start + offset] = as_i16(*value)?;
    }
    Ok(())
}

fn as_i16<T>(value: T) -> Result<i16>
where
    T: TryInto<i16>,
{
    value
        .try_into()
        .map_err(|_| invalid("relational token value does not fit signed 16-bit storage"))
}

#[cfg(test)]
mod tests {
    use cascadia_game::{GameConfig, GameSeed, GameState};
    use r3_action_edit_census::PublicStateTrunk;

    use super::*;

    #[test]
    fn rich_tokens_are_d6_complete_and_minimal_view_is_a_strict_projection() {
        let game = GameState::new(
            GameConfig::research_aaaaa(4).unwrap(),
            GameSeed::from_u64(31),
        )
        .unwrap();
        let trunk = PublicStateTrunk::observe(&game, 31).unwrap();
        let mut counts = Vec::new();
        for transform in D6Transform::ALL {
            let rich = rich_relational_parent_tokens(&trunk.sparse, transform).unwrap();
            assert!(
                rich.iter()
                    .any(|token| token.class_id == FRONTIER_SUMMARY_CLASS)
            );
            assert!(
                rich.iter()
                    .any(|token| token.class_id == OPPORTUNITY_SUMMARY_CLASS)
            );
            let minimal = rich.iter().filter_map(r5_minimal_token).collect::<Vec<_>>();
            assert!(!minimal.is_empty());
            assert!(minimal.len() < rich.len());
            assert!(
                minimal
                    .iter()
                    .all(|token| token.class_id <= R5_MINIMAL_CLASS_COUNT)
            );
            counts.push(rich.len());
        }
        assert!(counts.iter().all(|count| *count == counts[0]));
    }
}
