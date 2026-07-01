use std::collections::{BTreeMap, BTreeSet};

use blake3::Hasher;
use cascadia_data::{
    GRADED_SOURCE_BEST_CHAMPION_FRONTIER, GRADED_SOURCE_CHAMPION_SELECTED, GRADED_SOURCE_R600,
    GRADED_SOURCE_R1200, GRADED_SOURCE_R4800, GRADED_SOURCE_SENTINEL,
    GRADED_SOURCE_SUBSTANTIAL_TOP, GradedOracleCandidate, GradedOracleGroup,
};
use cascadia_game::D6Transform;
use r2_sparse_entity_census::{
    BOARD_TOKEN_CAPACITY, MlxEncodedState, TOKEN_PAYLOAD_WIDTH, transform_encoded_state,
};
use thiserror::Error;

pub const EXPERIMENT_ID: &str = "r3-action-edit-mlx-comparison-v1";
pub const PROTOCOL_ID: &str = "r3-action-edit-mlx-matched-comparison-v1";
pub const ADR_ID: &str = "0150";
pub const CACHE_SCHEMA: &str = "r3-action-edit-mlx-cache-v1";
pub const CACHE_SCHEMA_VERSION: u16 = 1;
pub const TRAIN_CANDIDATE_CAP: usize = 512;

const TRAIN_COHORT_DOMAIN: &[u8] = b"r3-mlx-train-cohort-v1";
const CONTROL_TOKEN_DOMAIN: &[u8] = b"r3-mlx-control-token-multiset-v1";

#[derive(Debug, Error)]
pub enum ExportError {
    #[error("R3 MLX export invariant failed: {0}")]
    Invariant(String),
    #[error(transparent)]
    R2(#[from] r2_sparse_entity_census::R2Error),
    #[error(transparent)]
    IntegerConversion(#[from] std::num::TryFromIntError),
}

pub type Result<T> = std::result::Result<T, ExportError>;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct R2TokenRecord {
    pub token_type: u8,
    pub payload: [i8; TOKEN_PAYLOAD_WIDTH],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ControlDelta {
    pub remove_indices: Vec<u8>,
    pub added: Vec<R2TokenRecord>,
    pub after_multiset_blake3: [u8; 32],
}

pub fn select_train_candidate_indices(group: &GradedOracleGroup) -> Result<Vec<usize>> {
    let count = group.candidates.len();
    if count <= TRAIN_CANDIDATE_CAP {
        return Ok((0..count).collect());
    }

    let mandatory_mask = GRADED_SOURCE_R4800
        | GRADED_SOURCE_R600
        | GRADED_SOURCE_SENTINEL
        | GRADED_SOURCE_SUBSTANTIAL_TOP
        | GRADED_SOURCE_BEST_CHAMPION_FRONTIER
        | GRADED_SOURCE_CHAMPION_SELECTED;
    let mut selected = BTreeSet::from([
        usize::from(group.selected_index),
        usize::from(group.champion_index),
    ]);
    for (index, candidate) in group.candidates.iter().enumerate() {
        if candidate.source_flags & mandatory_mask != 0 {
            selected.insert(index);
        }
    }
    if selected.len() > TRAIN_CANDIDATE_CAP {
        return Err(ExportError::Invariant(format!(
            "group {} has {} mandatory actions, above the 512-action cap",
            group.group_id,
            selected.len()
        )));
    }

    let mut screen_order = candidate_order(&group.candidates);
    for index in screen_order.iter().copied() {
        if selected.len() >= 256 {
            break;
        }
        selected.insert(index);
    }

    screen_order.retain(|index| {
        !selected.contains(index)
            && group.candidates[*index].source_flags & GRADED_SOURCE_R1200 != 0
    });
    let stratified_count = screen_order.len().min(128);
    for rank in 0..stratified_count {
        let index = if stratified_count == 1 {
            0
        } else {
            rank * (screen_order.len() - 1) / (stratified_count - 1)
        };
        selected.insert(screen_order[index]);
    }

    let mut remainder = (0..count)
        .filter(|index| !selected.contains(index))
        .map(|index| {
            let mut hasher = Hasher::new();
            hasher.update(TRAIN_COHORT_DOMAIN);
            hasher.update(&group.group_id.to_le_bytes());
            hasher.update(&group.candidates[index].action_hash);
            (*hasher.finalize().as_bytes(), index)
        })
        .collect::<Vec<_>>();
    remainder.sort_unstable();
    for (_, index) in remainder {
        if selected.len() >= TRAIN_CANDIDATE_CAP {
            break;
        }
        selected.insert(index);
    }

    if selected.len() != TRAIN_CANDIDATE_CAP {
        return Err(ExportError::Invariant(format!(
            "group {} retained {} actions instead of 512",
            group.group_id,
            selected.len()
        )));
    }
    Ok(selected.into_iter().collect())
}

pub fn cohort_blake3(group: &GradedOracleGroup, selected: &[usize]) -> Result<[u8; 32]> {
    if selected.windows(2).any(|pair| pair[0] >= pair[1])
        || selected
            .iter()
            .any(|index| *index >= group.candidates.len())
    {
        return Err(ExportError::Invariant(
            "train cohort indices are duplicated, unordered, or out of range".to_owned(),
        ));
    }
    let mut hasher = Hasher::new();
    hasher.update(TRAIN_COHORT_DOMAIN);
    hasher.update(&group.group_id.to_le_bytes());
    hasher.update(&(selected.len() as u64).to_le_bytes());
    for index in selected {
        hasher.update(&u16::try_from(*index)?.to_le_bytes());
        hasher.update(&group.candidates[*index].action_hash);
    }
    Ok(*hasher.finalize().as_bytes())
}

pub fn canonical_active_board_tokens(
    encoded: &MlxEncodedState,
    transform: D6Transform,
    transformed_center: (i8, i8),
) -> Result<Vec<R2TokenRecord>> {
    let transformed = transform_encoded_state(encoded, transform)?;
    active_board_tokens_relative(&transformed, transformed_center)
}

pub fn active_board_tokens_relative(
    transformed: &MlxEncodedState,
    transformed_center: (i8, i8),
) -> Result<Vec<R2TokenRecord>> {
    let active = transformed.board_type_counts[0]
        .iter()
        .map(|value| usize::from(*value))
        .sum::<usize>();
    if active > BOARD_TOKEN_CAPACITY {
        return Err(ExportError::Invariant(
            "active R2 board exceeds its 92-token capacity".to_owned(),
        ));
    }
    let mut records = Vec::with_capacity(active);
    for slot in 0..active {
        let token_type = transformed.token_types[slot];
        if !(1..=4).contains(&token_type) {
            return Err(ExportError::Invariant(
                "active R2 board contains an invalid token type".to_owned(),
            ));
        }
        let start = slot * TOKEN_PAYLOAD_WIDTH;
        let mut payload = [0i8; TOKEN_PAYLOAD_WIDTH];
        payload.copy_from_slice(&transformed.token_payload[start..start + TOKEN_PAYLOAD_WIDTH]);
        translate_payload(&mut payload, token_type, transformed_center)?;
        records.push(R2TokenRecord {
            token_type,
            payload,
        });
    }
    Ok(records)
}

pub fn control_delta(parent: &[R2TokenRecord], after: &[R2TokenRecord]) -> Result<ControlDelta> {
    let mut available = BTreeMap::<R2TokenRecord, Vec<usize>>::new();
    for (index, token) in parent.iter().cloned().enumerate() {
        available.entry(token).or_default().push(index);
    }
    let mut added = Vec::new();
    for token in after {
        match available.get_mut(token).and_then(Vec::pop) {
            Some(_) => {}
            None => added.push(token.clone()),
        }
    }
    let mut remove_indices = available.into_values().flatten().collect::<Vec<_>>();
    remove_indices.sort_unstable();
    let remove_set = remove_indices.iter().copied().collect::<BTreeSet<_>>();
    let mut reconstructed = parent
        .iter()
        .enumerate()
        .filter(|(index, _)| !remove_set.contains(index))
        .map(|(_, token)| token.clone())
        .collect::<Vec<_>>();
    reconstructed.extend(added.iter().cloned());
    let mut expected = after.to_vec();
    reconstructed.sort_unstable();
    expected.sort_unstable();
    if reconstructed != expected {
        return Err(ExportError::Invariant(
            "control parent delta does not reconstruct the exact afterstate multiset".to_owned(),
        ));
    }
    Ok(ControlDelta {
        remove_indices: remove_indices
            .into_iter()
            .map(u8::try_from)
            .collect::<std::result::Result<_, _>>()?,
        added,
        after_multiset_blake3: token_multiset_blake3(after),
    })
}

pub fn token_multiset_blake3(tokens: &[R2TokenRecord]) -> [u8; 32] {
    let mut sorted = tokens.to_vec();
    sorted.sort_unstable();
    let mut hasher = Hasher::new();
    hasher.update(CONTROL_TOKEN_DOMAIN);
    hasher.update(&(sorted.len() as u64).to_le_bytes());
    for token in sorted {
        hasher.update(&[token.token_type]);
        hasher.update(
            &token
                .payload
                .into_iter()
                .map(|value| value as u8)
                .collect::<Vec<_>>(),
        );
    }
    *hasher.finalize().as_bytes()
}

fn candidate_order(candidates: &[GradedOracleCandidate]) -> Vec<usize> {
    let mut order = (0..candidates.len()).collect::<Vec<_>>();
    order.sort_unstable_by_key(|index| {
        (
            candidates[*index].screen_rank,
            candidates[*index].action_hash,
        )
    });
    order
}

fn translate_payload(
    payload: &mut [i8; TOKEN_PAYLOAD_WIDTH],
    token_type: u8,
    center: (i8, i8),
) -> Result<()> {
    match token_type {
        1 | 2 | 4 => translate_coord(payload, 0, 1, center)?,
        3 => {
            let member_count = usize::try_from(payload[2]).map_err(|_| {
                ExportError::Invariant("R2 component member count is negative".to_owned())
            })?;
            if member_count > 23 || 6 + member_count * 2 > TOKEN_PAYLOAD_WIDTH {
                return Err(ExportError::Invariant(
                    "R2 component member payload exceeds the board bound".to_owned(),
                ));
            }
            for member in 0..member_count {
                translate_coord(payload, 6 + member * 2, 7 + member * 2, center)?;
            }
        }
        _ => {
            return Err(ExportError::Invariant(format!(
                "cannot translate unknown R2 token type {token_type}"
            )));
        }
    }
    Ok(())
}

fn translate_coord(
    payload: &mut [i8; TOKEN_PAYLOAD_WIDTH],
    q_slot: usize,
    r_slot: usize,
    center: (i8, i8),
) -> Result<()> {
    let q = i16::from(payload[q_slot]) - i16::from(center.0);
    let r = i16::from(payload[r_slot]) - i16::from(center.1);
    payload[q_slot] = i8::try_from(q).map_err(|_| {
        ExportError::Invariant("canonical relative q coordinate exceeds i8".to_owned())
    })?;
    payload[r_slot] = i8::try_from(r).map_err(|_| {
        ExportError::Invariant("canonical relative r coordinate exceeds i8".to_owned())
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use cascadia_data::{GRADED_ORACLE_CANDIDATE_RECORD_SIZE, GRADED_SOURCE_COMPLETE_LEGAL};

    fn candidate(index: usize, flags: u16) -> GradedOracleCandidate {
        let mut candidate =
            GradedOracleCandidate::from_bytes(&[0; GRADED_ORACLE_CANDIDATE_RECORD_SIZE]);
        candidate.action_hash = *blake3::hash(&(index as u64).to_le_bytes()).as_bytes();
        candidate.canonical_index = index as u16;
        candidate.screen_rank = (index + 1) as u16;
        candidate.source_flags = flags | GRADED_SOURCE_COMPLETE_LEGAL;
        candidate
    }

    #[test]
    fn control_delta_reconstructs_duplicate_token_multisets() {
        let token = R2TokenRecord {
            token_type: 1,
            payload: [0; TOKEN_PAYLOAD_WIDTH],
        };
        let changed = R2TokenRecord {
            token_type: 4,
            payload: [1; TOKEN_PAYLOAD_WIDTH],
        };
        let delta = control_delta(
            &[token.clone(), token.clone()],
            &[token.clone(), changed.clone()],
        )
        .unwrap();
        assert_eq!(delta.remove_indices.len(), 1);
        assert_eq!(delta.added, vec![changed]);
    }

    #[test]
    fn candidate_order_is_stable_by_rank_then_hash() {
        let mut candidates = vec![candidate(0, 0), candidate(1, 0)];
        candidates[0].screen_rank = 2;
        candidates[1].screen_rank = 1;
        assert_eq!(candidate_order(&candidates), vec![1, 0]);
    }
}
