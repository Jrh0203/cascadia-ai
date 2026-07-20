use std::collections::{HashMap, HashSet};

use anyhow::{Context, Result, bail};
use cascadia_game::{GRID_DIM, ScoringCards, ScoringVariant};
use half::f16;
use serde_json::Value;

pub const SHARD_VERSION: &str = "greedy_policy_tensor_shard_v1";
pub const EXPERT_SHARD_VERSION: &str = "cascadiav3.expert_tensor_shard.v1";
pub const EXPERT_SHARD_VERSION_V4: &str = "cascadiav3.expert_tensor_shard.v4";
pub const PUBLIC_TOKEN_FEATURE_DIM: usize = 41;
pub const MERIT_ACTION_FEATURE_DIM: usize = 25;
pub const PUBLIC_TOKEN_ACTION_FEATURE_DIM: usize = 33;
pub const SEMANTIC_ACTION_FEATURE_DIM: usize = 28;
pub const SEMANTIC_PUBLIC_TOKEN_ACTION_FEATURE_DIM: usize =
    PUBLIC_TOKEN_ACTION_FEATURE_DIM + SEMANTIC_ACTION_FEATURE_DIM;

const TOKEN_KINDS: [&str; 6] = [
    "player",
    "placed_tile",
    "frontier",
    "market_tile",
    "market_wildlife",
    "public_supply",
];
const SPECIES_NAMES: [&str; 5] = ["Bear", "Elk", "Salmon", "Hawk", "Fox"];
const DIRECTIONS: [(i32, i32); 6] = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)];
const AXIS_DIRECTIONS: [(i32, i32); 3] = [(1, 0), (1, -1), (0, -1)];
const WILDLIFE_COUNT: usize = 5;

#[derive(Debug, Clone, Default)]
pub struct TensorShardData {
    pub tokens_f16_bits: Vec<u16>,
    pub actions_f16_bits: Vec<u16>,
    pub token_offsets: Vec<i64>,
    pub action_offsets: Vec<i64>,
    pub selected_action_index: Vec<i16>,
    pub record_count: usize,
    pub total_token_count: usize,
    pub total_action_count: usize,
    pub max_token_count: usize,
    pub max_action_count: usize,
    pub first_state_hash: Option<String>,
    pub last_state_hash: Option<String>,
    /// Active scoring-card ruleset for this shard's card-aware semantic
    /// features. Defaults to AAAAA (byte-identical to legacy output).
    pub scoring_cards: ScoringCards,
}

#[derive(Debug, Clone, Default)]
pub struct ExpertTensorShardData {
    pub tokens_f16_bits: Vec<u16>,
    pub actions_f16_bits: Vec<u16>,
    pub token_offsets: Vec<i64>,
    pub action_offsets: Vec<i64>,
    pub relation_edges_i32: Vec<i32>,
    pub relation_offsets: Vec<i64>,
    pub selected_action_index: Vec<i16>,
    pub target_q: Vec<f32>,
    pub target_score_to_go: Vec<f32>,
    pub q_valid: Vec<u8>,
    pub priors: Vec<f32>,
    pub visits: Vec<f32>,
    pub q_variance: Vec<f32>,
    pub q_count: Vec<f32>,
    pub truncated_count: Vec<f32>,
    pub exact_afterstate_score_active: Vec<f32>,
    /// v4 action-aligned [wildlife, habitat, nature_tokens] exact score.
    pub exact_afterstate_score_decomposition_active: Vec<f32>,
    /// v4 explicit active seat per root.
    pub active_seat: Vec<u8>,
    pub structured_value_field_records: usize,
    pub final_score_vector: Vec<f32>,
    pub rank_vector: Vec<i16>,
    pub score_decomposition: Vec<f32>,
    /// v2 fields: action-aligned improved policy targets and per-record search
    /// root values. Present only when every record carries them.
    pub improved_policy: Vec<f32>,
    pub search_root_value: Vec<f32>,
    pub improved_policy_records: usize,
    /// v3 field: one explicit boolean per root. This must not be inferred
    /// from visits or policy shape because exact K1 and sampled roots can
    /// otherwise be indistinguishable after packing.
    pub exact_endgame: Vec<u8>,
    pub exact_endgame_field_records: usize,
    pub record_count: usize,
    pub total_token_count: usize,
    pub total_action_count: usize,
    pub total_relation_edge_count: usize,
    pub max_token_count: usize,
    pub max_action_count: usize,
    pub max_relation_edge_count: usize,
    pub first_state_hash: Option<String>,
    pub last_state_hash: Option<String>,
    /// Active scoring-card ruleset for this shard. Card-aware feature builders
    /// (semantic action hints + hawk line-of-sight relation edges) key off this
    /// so a CBDDB shard is scored-matched while an AAAAA shard is byte-identical
    /// to the legacy Card-A output. Defaults to AAAAA.
    pub scoring_cards: ScoringCards,
}

impl ExpertTensorShardData {
    pub fn new() -> Self {
        Self {
            token_offsets: vec![0],
            action_offsets: vec![0],
            relation_offsets: vec![0],
            ..Self::default()
        }
    }

    /// Aggregator variant that stamps the active scoring-card ruleset so every
    /// record packed into this shard uses card-matched features.
    pub fn with_scoring_cards(cards: ScoringCards) -> Self {
        Self {
            scoring_cards: cards,
            ..Self::new()
        }
    }

    /// AAAAA convenience constructor retained for the byte-identity golden
    /// tests; production always routes through `from_records_with_cards`.
    #[cfg(test)]
    pub fn from_records(records: &[Value]) -> Result<Self> {
        Self::from_records_with_cards(records, ScoringCards::AAAAA)
    }

    pub fn from_records_with_cards(records: &[Value], cards: ScoringCards) -> Result<Self> {
        let mut data = Self::with_scoring_cards(cards);
        for record in records {
            data.push_record(record)?;
        }
        Ok(data)
    }

    pub fn merge(&mut self, other: Self) {
        if other.record_count == 0 {
            return;
        }
        // Every shard in a run shares the run's ruleset; adopt it from the
        // first non-empty contributor and assert the rest agree so an AAAAA and
        // a CBDDB shard can never be silently concatenated.
        if self.record_count == 0 {
            self.scoring_cards = other.scoring_cards;
        } else {
            assert_eq!(
                self.scoring_cards, other.scoring_cards,
                "cannot merge shards built under different scoring-card rulesets",
            );
        }
        if self.first_state_hash.is_none() {
            self.first_state_hash = other.first_state_hash.clone();
        }
        self.last_state_hash = other.last_state_hash.clone();
        let token_base = self.total_token_count as i64;
        let action_base = self.total_action_count as i64;
        let relation_base = self.total_relation_edge_count as i64;
        self.tokens_f16_bits.extend(other.tokens_f16_bits);
        self.actions_f16_bits.extend(other.actions_f16_bits);
        self.relation_edges_i32.extend(other.relation_edges_i32);
        self.token_offsets.extend(
            other
                .token_offsets
                .iter()
                .skip(1)
                .map(|offset| token_base + *offset),
        );
        self.action_offsets.extend(
            other
                .action_offsets
                .iter()
                .skip(1)
                .map(|offset| action_base + *offset),
        );
        self.relation_offsets.extend(
            other
                .relation_offsets
                .iter()
                .skip(1)
                .map(|offset| relation_base + *offset),
        );
        self.selected_action_index
            .extend(other.selected_action_index);
        self.target_q.extend(other.target_q);
        self.target_score_to_go.extend(other.target_score_to_go);
        self.q_valid.extend(other.q_valid);
        self.priors.extend(other.priors);
        self.visits.extend(other.visits);
        self.q_variance.extend(other.q_variance);
        self.q_count.extend(other.q_count);
        self.truncated_count.extend(other.truncated_count);
        self.exact_afterstate_score_active
            .extend(other.exact_afterstate_score_active);
        self.exact_afterstate_score_decomposition_active
            .extend(other.exact_afterstate_score_decomposition_active);
        self.active_seat.extend(other.active_seat);
        self.structured_value_field_records += other.structured_value_field_records;
        self.final_score_vector.extend(other.final_score_vector);
        self.rank_vector.extend(other.rank_vector);
        self.score_decomposition.extend(other.score_decomposition);
        self.improved_policy.extend(other.improved_policy);
        self.search_root_value.extend(other.search_root_value);
        self.improved_policy_records += other.improved_policy_records;
        self.exact_endgame.extend(other.exact_endgame);
        self.exact_endgame_field_records += other.exact_endgame_field_records;
        self.record_count += other.record_count;
        self.total_token_count += other.total_token_count;
        self.total_action_count += other.total_action_count;
        self.total_relation_edge_count += other.total_relation_edge_count;
        self.max_token_count = self.max_token_count.max(other.max_token_count);
        self.max_action_count = self.max_action_count.max(other.max_action_count);
        self.max_relation_edge_count = self
            .max_relation_edge_count
            .max(other.max_relation_edge_count);
    }

    fn push_record(&mut self, record: &Value) -> Result<()> {
        let token_rows = public_token_features(record)?;
        let action_rows = semantic_public_token_action_features(record, self.scoring_cards)?;
        let action_count = action_rows.len();
        if action_count == 0 {
            bail!("expert tensor record has no legal actions");
        }
        let selected = selected_action_index(record)?;
        if selected >= action_count {
            bail!("selected action index exceeds legal action count");
        }
        let relation_edges =
            combined_relation_edges(record, token_rows.len(), action_count, self.scoring_cards)?;

        let target_q = f32_array(record, "per_action_Q", action_count)?;
        let score_to_go = f32_array(record, "per_action_score_to_go", action_count)?;
        let priors = f32_array(record, "priors", action_count)?;
        let visits = f32_array(record, "visits", action_count)?;
        let q_variance = f32_array(record, "per_action_Q_variance", action_count)?;
        let q_count = f32_array(record, "per_action_Q_count", action_count)?;
        let truncated = f32_array(record, "per_action_truncated_count", action_count)?;
        let exact_afterstate = f32_array(record, "exact_afterstate_score_active", action_count)?;
        let q_valid = bool_array(record, "per_action_Q_valid", action_count)?;
        let final_score = f32_array(record, "final_score_vector", 4)?;
        let rank = i16_array(record, "rank_vector", 4)?;
        let score_decomposition = score_decomposition_array(record)?;
        let structured_value_fields = match record
            .get("exact_afterstate_score_decomposition_active")
        {
            Some(_) => {
                let decomposition = action_score_decomposition_array(
                    record,
                    "exact_afterstate_score_decomposition_active",
                    action_count,
                )?;
                let active_seat = record
                    .get("active_seat")
                    .context("structured expert tensor record is missing active_seat")?
                    .as_u64()
                    .context("expert tensor active_seat must be an integer")?;
                if active_seat >= 4 {
                    bail!("expert tensor active_seat must be in [0, 4)");
                }
                for (index, components) in decomposition.chunks_exact(3).enumerate() {
                    let component_total: f32 = components.iter().sum();
                    if (component_total - exact_afterstate[index]).abs() > 1.0e-4 {
                        bail!(
                            "exact afterstate component sum mismatch at action {index}: {component_total} != {}",
                            exact_afterstate[index]
                        );
                    }
                }
                for seat in 0..4 {
                    let component_total = (0..3)
                        .map(|category| score_decomposition[category * 4 + seat])
                        .sum::<f32>();
                    if (component_total - final_score[seat]).abs() > 1.0e-4 {
                        bail!(
                            "terminal component sum mismatch at seat {seat}: {component_total} != {}",
                            final_score[seat]
                        );
                    }
                }
                Some((decomposition, active_seat as u8))
            }
            None => None,
        };

        if self.first_state_hash.is_none() {
            self.first_state_hash = string_field(record, "state_hash");
        }
        self.last_state_hash = string_field(record, "state_hash");

        for value in token_rows.iter().flatten() {
            self.tokens_f16_bits.push(f16::from_f32(*value).to_bits());
        }
        for value in action_rows.iter().flatten() {
            self.actions_f16_bits.push(f16::from_f32(*value).to_bits());
        }
        for [source, target, relation_id] in &relation_edges {
            self.relation_edges_i32.push(*source);
            self.relation_edges_i32.push(*target);
            self.relation_edges_i32.push(*relation_id);
        }
        self.selected_action_index.push(selected as i16);
        self.target_q.extend(target_q);
        self.target_score_to_go.extend(score_to_go);
        self.q_valid.extend(q_valid);
        self.priors.extend(priors);
        self.visits.extend(visits);
        self.q_variance.extend(q_variance);
        self.q_count.extend(q_count);
        self.truncated_count.extend(truncated);
        self.exact_afterstate_score_active.extend(exact_afterstate);
        if let Some((decomposition, active_seat)) = structured_value_fields {
            self.exact_afterstate_score_decomposition_active
                .extend(decomposition);
            self.active_seat.push(active_seat);
            self.structured_value_field_records += 1;
        }
        self.final_score_vector.extend(final_score);
        self.rank_vector.extend(rank);
        self.score_decomposition.extend(score_decomposition);
        if record.get("improved_policy").is_some() {
            let improved = f32_array(record, "improved_policy", action_count)?;
            let root_value = record
                .get("search_root_value")
                .and_then(Value::as_f64)
                .context("record with improved_policy is missing search_root_value")?;
            self.improved_policy.extend(improved);
            self.search_root_value.push(root_value as f32);
            self.improved_policy_records += 1;
        }
        if let Some(exact_endgame) = record.get("exact_endgame").and_then(Value::as_bool) {
            self.exact_endgame.push(u8::from(exact_endgame));
            self.exact_endgame_field_records += 1;
        }

        self.record_count += 1;
        self.total_token_count += token_rows.len();
        self.total_action_count += action_rows.len();
        self.total_relation_edge_count += relation_edges.len();
        self.max_token_count = self.max_token_count.max(token_rows.len());
        self.max_action_count = self.max_action_count.max(action_count);
        self.max_relation_edge_count = self.max_relation_edge_count.max(relation_edges.len());
        self.token_offsets.push(self.total_token_count as i64);
        self.action_offsets.push(self.total_action_count as i64);
        self.relation_offsets
            .push(self.total_relation_edge_count as i64);
        Ok(())
    }
}

impl TensorShardData {
    pub fn new() -> Self {
        Self {
            token_offsets: vec![0],
            action_offsets: vec![0],
            ..Self::default()
        }
    }

    /// Aggregator variant that stamps the active scoring-card ruleset so every
    /// record packed into this shard uses card-matched semantic features.
    pub fn with_scoring_cards(cards: ScoringCards) -> Self {
        Self {
            scoring_cards: cards,
            ..Self::new()
        }
    }

    pub fn from_records_with_cards(records: &[Value], cards: ScoringCards) -> Result<Self> {
        let mut data = Self::with_scoring_cards(cards);
        for record in records {
            data.push_record(record)?;
        }
        Ok(data)
    }

    pub fn merge(&mut self, other: Self) {
        if other.record_count == 0 {
            return;
        }
        if self.record_count == 0 {
            self.scoring_cards = other.scoring_cards;
        } else {
            assert_eq!(
                self.scoring_cards, other.scoring_cards,
                "cannot merge shards built under different scoring-card rulesets",
            );
        }
        if self.first_state_hash.is_none() {
            self.first_state_hash = other.first_state_hash.clone();
        }
        self.last_state_hash = other.last_state_hash.clone();
        let token_base = self.total_token_count as i64;
        let action_base = self.total_action_count as i64;
        self.tokens_f16_bits.extend(other.tokens_f16_bits);
        self.actions_f16_bits.extend(other.actions_f16_bits);
        self.token_offsets.extend(
            other
                .token_offsets
                .iter()
                .skip(1)
                .map(|offset| token_base + *offset),
        );
        self.action_offsets.extend(
            other
                .action_offsets
                .iter()
                .skip(1)
                .map(|offset| action_base + *offset),
        );
        self.selected_action_index
            .extend(other.selected_action_index);
        self.record_count += other.record_count;
        self.total_token_count += other.total_token_count;
        self.total_action_count += other.total_action_count;
        self.max_token_count = self.max_token_count.max(other.max_token_count);
        self.max_action_count = self.max_action_count.max(other.max_action_count);
    }

    fn push_record(&mut self, record: &Value) -> Result<()> {
        let token_rows = public_token_features(record)?;
        let action_rows = semantic_public_token_action_features(record, self.scoring_cards)?;
        let selected = selected_action_index(record)?;
        if selected >= action_rows.len() {
            bail!("selected action index exceeds legal action count");
        }
        if self.first_state_hash.is_none() {
            self.first_state_hash = string_field(record, "state_hash");
        }
        self.last_state_hash = string_field(record, "state_hash");
        for value in token_rows.iter().flatten() {
            self.tokens_f16_bits.push(f16::from_f32(*value).to_bits());
        }
        for value in action_rows.iter().flatten() {
            self.actions_f16_bits.push(f16::from_f32(*value).to_bits());
        }
        self.record_count += 1;
        self.total_token_count += token_rows.len();
        self.total_action_count += action_rows.len();
        self.max_token_count = self.max_token_count.max(token_rows.len());
        self.max_action_count = self.max_action_count.max(action_rows.len());
        self.token_offsets.push(self.total_token_count as i64);
        self.action_offsets.push(self.total_action_count as i64);
        self.selected_action_index.push(selected as i16);
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Coord {
    q: i32,
    r: i32,
}

impl Coord {
    fn neighbor(self, direction: (i32, i32)) -> Self {
        Self {
            q: self.q + direction.0,
            r: self.r + direction.1,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct DegreeSummary {
    adjacent_out: f32,
    adjacent_in: f32,
    terrain_match_out: f32,
    market_pair: f32,
}

#[derive(Debug, Clone, Copy)]
struct TileInfo {
    terrain_a: i32,
    terrain_b: i32,
    rotation: i32,
}

#[derive(Debug, Default)]
struct StateView {
    active_seat: i64,
    active_tiles: HashMap<Coord, TileInfo>,
    wildlife_by_owner: HashMap<i64, HashMap<i64, HashSet<Coord>>>,
    active_wildlife: HashMap<i64, HashSet<Coord>>,
    empty_species_slots: [i32; WILDLIFE_COUNT],
    market_species_counts: [i32; WILDLIFE_COUNT],
    supply_bag: [f32; WILDLIFE_COUNT],
    supply_capacity: [f32; WILDLIFE_COUNT],
}

fn field<'a>(value: &'a Value, key: &str) -> Option<&'a Value> {
    value.as_object().and_then(|object| object.get(key))
}

fn string_field(value: &Value, key: &str) -> Option<String> {
    field(value, key).and_then(Value::as_str).map(str::to_owned)
}

fn safe_f64(value: Option<&Value>, default: f32) -> f32 {
    match value {
        Some(Value::Number(number)) => number.as_f64().unwrap_or(default as f64) as f32,
        Some(Value::String(text)) => text.parse::<f32>().unwrap_or(default),
        Some(Value::Bool(flag)) => {
            if *flag {
                1.0
            } else {
                0.0
            }
        }
        _ => default,
    }
}

fn safe_i64(value: Option<&Value>, default: i64) -> i64 {
    safe_f64(value, default as f32) as i64
}

fn normalizer(value: f32, scale: f32) -> f32 {
    if scale == 0.0 { value } else { value / scale }
}

fn bool_field(value: &Value, key: &str) -> bool {
    field(value, key).and_then(Value::as_bool).unwrap_or(false)
}

fn f32_array(record: &Value, key: &str, expected_len: usize) -> Result<Vec<f32>> {
    let values = field(record, key)
        .and_then(Value::as_array)
        .with_context(|| format!("expert tensor record missing array {key}"))?;
    if values.len() != expected_len {
        bail!(
            "expert tensor array {key} length {} does not match expected {expected_len}",
            values.len()
        );
    }
    values
        .iter()
        .map(|value| {
            value
                .as_f64()
                .map(|number| number as f32)
                .with_context(|| format!("expert tensor array {key} contains non-number"))
        })
        .collect()
}

fn action_score_decomposition_array(
    record: &Value,
    key: &str,
    action_count: usize,
) -> Result<Vec<f32>> {
    let rows = field(record, key)
        .and_then(Value::as_array)
        .with_context(|| format!("expert tensor record missing array {key}"))?;
    if rows.len() != action_count {
        bail!(
            "expert tensor array {key} length {} does not match expected {action_count}",
            rows.len()
        );
    }
    let mut flattened = Vec::with_capacity(action_count * 3);
    for (index, row) in rows.iter().enumerate() {
        let values = row
            .as_array()
            .with_context(|| format!("expert tensor array {key}[{index}] is not an array"))?;
        if values.len() != 3 {
            bail!("expert tensor array {key}[{index}] must have three components");
        }
        for value in values {
            let number = value.as_f64().with_context(|| {
                format!("expert tensor array {key}[{index}] contains non-number")
            })? as f32;
            if !number.is_finite() {
                bail!("expert tensor array {key}[{index}] contains non-finite value");
            }
            flattened.push(number);
        }
    }
    Ok(flattened)
}

fn i16_array(record: &Value, key: &str, expected_len: usize) -> Result<Vec<i16>> {
    let values = field(record, key)
        .and_then(Value::as_array)
        .with_context(|| format!("expert tensor record missing array {key}"))?;
    if values.len() != expected_len {
        bail!(
            "expert tensor array {key} length {} does not match expected {expected_len}",
            values.len()
        );
    }
    values
        .iter()
        .map(|value| {
            value
                .as_i64()
                .map(|number| number as i16)
                .with_context(|| format!("expert tensor array {key} contains non-integer"))
        })
        .collect()
}

fn bool_array(record: &Value, key: &str, expected_len: usize) -> Result<Vec<u8>> {
    let values = field(record, key)
        .and_then(Value::as_array)
        .with_context(|| format!("expert tensor record missing array {key}"))?;
    if values.len() != expected_len {
        bail!(
            "expert tensor array {key} length {} does not match expected {expected_len}",
            values.len()
        );
    }
    values
        .iter()
        .map(|value| {
            value
                .as_bool()
                .map(u8::from)
                .with_context(|| format!("expert tensor array {key} contains non-bool"))
        })
        .collect()
}

fn score_decomposition_array(record: &Value) -> Result<Vec<f32>> {
    let object = field(record, "score_decomposition")
        .and_then(Value::as_object)
        .context("expert tensor record missing score_decomposition")?;
    let mut out = Vec::with_capacity(12);
    for category in ["wildlife", "habitat", "nature_tokens"] {
        for seat in 0..4 {
            let parts = object
                .get(&seat.to_string())
                .and_then(Value::as_object)
                .with_context(|| format!("score_decomposition missing seat {seat}"))?;
            let value = parts
                .get(category)
                .and_then(Value::as_f64)
                .with_context(|| format!("score_decomposition seat {seat} missing {category}"))?;
            out.push(value as f32);
        }
    }
    Ok(out)
}

fn coord_key(value: Option<&Value>) -> Option<Coord> {
    let coord = value?.as_object()?;
    Some(Coord {
        q: safe_i64(coord.get("q"), 0) as i32,
        r: safe_i64(coord.get("r"), 0) as i32,
    })
}

fn relation_coord_key(value: Option<&Value>) -> Option<String> {
    let coord = value?.as_object()?;
    match coord.get("kind").and_then(Value::as_str) {
        Some("canonical") => Some(format!(
            "canonical:{}",
            safe_i64(coord.get("cell_index"), -1)
        )),
        _ => Some(format!(
            "overflow:{}:{}:{}:{}",
            safe_i64(coord.get("owner_seat"), -1),
            safe_i64(coord.get("placement_id"), -1),
            safe_i64(coord.get("q"), 0),
            safe_i64(coord.get("r"), 0)
        )),
    }
}

fn set_relation(
    edges: &mut HashMap<(i32, i32), i32>,
    source: i32,
    target: i32,
    relation_id: i32,
    overwrite: bool,
    seq_len: i32,
) {
    if source < 0 || target < 0 || source >= seq_len || target >= seq_len || source == target {
        return;
    }
    if overwrite || !edges.contains_key(&(source, target)) {
        edges.insert((source, target), relation_id);
    }
}

fn combined_relation_edges(
    root: &Value,
    token_count: usize,
    action_count: usize,
    cards: ScoringCards,
) -> Result<Vec<[i32; 3]>> {
    const ADJACENT_HEX: i32 = 1;
    const TERRAIN_MATCH_ADJACENT: i32 = 2;
    const SAME_MARKET_SLOT: i32 = 3;
    const SAME_OWNER_BOARD: i32 = 4;
    const ACTION_USES_TILE_SLOT: i32 = 5;
    const ACTION_USES_WILDLIFE_SLOT: i32 = 6;
    const ACTION_TARGETS_TILE_FRONTIER: i32 = 7;
    const ACTION_TARGETS_WILDLIFE_CELL: i32 = 8;

    let seq_len = (token_count + action_count) as i32;
    let mut edges: HashMap<(i32, i32), i32> = HashMap::new();
    let tokens = field(root, "public_tokens")
        .and_then(|public_tokens| field(public_tokens, "tokens"))
        .and_then(Value::as_array)
        .context("root is missing public tokens")?;
    let active_seat = safe_i64(field(root, "active_seat"), 0);
    let mut tokens_by_owner: HashMap<i64, Vec<i32>> = HashMap::new();
    let mut market_tile: HashMap<i64, i32> = HashMap::new();
    let mut market_wildlife: HashMap<i64, i32> = HashMap::new();
    let mut active_frontier: HashMap<String, i32> = HashMap::new();
    let mut active_tile: HashMap<String, i32> = HashMap::new();

    for token in tokens {
        let index = safe_i64(field(token, "token_index"), -1) as i32;
        let kind = field(token, "token_kind").and_then(Value::as_str);
        if let Some(owner) = field(token, "owner_seat").and_then(Value::as_i64) {
            if matches!(kind, Some("player" | "placed_tile" | "frontier")) {
                tokens_by_owner.entry(owner).or_default().push(index);
            }
        }
        match kind {
            Some("market_tile") => {
                market_tile.insert(safe_i64(field(token, "market_slot"), -1), index);
            }
            Some("market_wildlife") => {
                market_wildlife.insert(safe_i64(field(token, "market_slot"), -1), index);
            }
            Some("frontier") if safe_i64(field(token, "owner_seat"), -1) == active_seat => {
                if let Some(key) = relation_coord_key(field(token, "coord_ref")) {
                    active_frontier.insert(key, index);
                }
            }
            Some("placed_tile") if safe_i64(field(token, "owner_seat"), -1) == active_seat => {
                if let Some(key) = relation_coord_key(field(token, "coord_ref")) {
                    active_tile.insert(key, index);
                }
            }
            _ => {}
        }
    }

    for indexes in tokens_by_owner.values() {
        for source in indexes {
            for target in indexes {
                set_relation(
                    &mut edges,
                    *source,
                    *target,
                    SAME_OWNER_BOARD,
                    false,
                    seq_len,
                );
            }
        }
    }

    let relations = field(root, "public_tokens")
        .and_then(|public_tokens| field(public_tokens, "relations"))
        .and_then(Value::as_array)
        .context("root is missing public token relations")?;
    for relation in relations {
        let source = safe_i64(field(relation, "source"), -1) as i32;
        let target = safe_i64(field(relation, "target"), -1) as i32;
        match field(relation, "relation_kind").and_then(Value::as_str) {
            Some("adjacent_hex") => {
                let relation_id = if bool_field(relation, "terrain_matches") {
                    TERRAIN_MATCH_ADJACENT
                } else {
                    ADJACENT_HEX
                };
                set_relation(&mut edges, source, target, relation_id, true, seq_len);
            }
            Some("same_market_slot") => {
                set_relation(&mut edges, source, target, SAME_MARKET_SLOT, true, seq_len);
            }
            _ => {}
        }
    }

    let actions = field(root, "legal_actions")
        .and_then(Value::as_array)
        .context("root is missing legal_actions")?;
    for (action_index, action) in actions.iter().enumerate() {
        let action_pos = token_count as i32 + action_index as i32;
        let tile_slot = safe_i64(
            field(action, "tile_slot").or_else(|| field(action, "draft_slot")),
            -1,
        );
        let wildlife_slot = safe_i64(
            field(action, "wildlife_slot").or_else(|| field(action, "draft_slot")),
            -1,
        );
        let target_frontier = relation_coord_key(field(action, "target_coord_ref"))
            .and_then(|key| active_frontier.get(&key).copied());
        let wildlife_target =
            relation_coord_key(field(action, "wildlife_coord_ref")).and_then(|key| {
                active_tile
                    .get(&key)
                    .or_else(|| active_frontier.get(&key))
                    .copied()
            });
        for (target, relation_id) in [
            (market_tile.get(&tile_slot).copied(), ACTION_USES_TILE_SLOT),
            (
                market_wildlife.get(&wildlife_slot).copied(),
                ACTION_USES_WILDLIFE_SLOT,
            ),
            (target_frontier, ACTION_TARGETS_TILE_FRONTIER),
            (wildlife_target, ACTION_TARGETS_WILDLIFE_CELL),
        ] {
            let Some(target) = target else {
                continue;
            };
            set_relation(&mut edges, action_pos, target, relation_id, true, seq_len);
            set_relation(&mut edges, target, action_pos, relation_id, true, seq_len);
        }
    }

    // Hawk line-of-sight edges (Hawk C/D only). Under Hawk A this block is
    // skipped entirely, so the AAAAA edge set is byte-identical. The
    // between-species count is bucketed into distinct relation ids so the model
    // sees the Hawk-D weight tier (0/4/7/9) directly. These are token->token
    // edges (both endpoints are placed hawk tiles); they overwrite the generic
    // SAME_OWNER_BOARD edge for that specific pair because line-of-sight is the
    // strictly more informative relation.
    if matches!(cards.hawk, ScoringVariant::C | ScoringVariant::D) {
        for (source, target, relation_id) in hawk_line_of_sight_relation_edges(tokens) {
            set_relation(&mut edges, source, target, relation_id, true, seq_len);
            set_relation(&mut edges, target, source, relation_id, true, seq_len);
        }
    }

    // Action-source hawk line-of-sight edges (Hawk C/D only), consumed by the
    // model's gated action bias. Emitted AFTER the ids-5..8 action loop above so
    // that, on the rare cell where an action row already points at the partner
    // token, the more-informative LOS id wins under the same last-write-wins
    // overwrite convention (matched byte-for-byte by `action_relation_tail`).
    for (action_index, partner_node, relation_id) in
        hawk_los_action_edges(root, active_seat, cards)
    {
        let action_pos = token_count as i32 + action_index as i32;
        set_relation(&mut edges, action_pos, partner_node, relation_id, true, seq_len);
        set_relation(&mut edges, partner_node, action_pos, relation_id, true, seq_len);
    }

    let mut out = edges
        .into_iter()
        .filter_map(|((source, target), relation_id)| {
            (relation_id != 0).then_some([source, target, relation_id])
        })
        .collect::<Vec<_>>();
    out.sort_unstable_by_key(|edge| (edge[0], edge[1], edge[2]));
    Ok(out)
}

/// Line-of-sight relation edges among placed hawk tokens, grouped per owner
/// board (a hawk can only see hawks on its own board). Each returned tuple is
/// `(source_node, target_node, relation_id)` where the id encodes the count of
/// DISTINCT non-hawk species strictly between the pair, mirroring the engine's
/// `hawk_line_of_sight_pairs` (see crates/cascadia-game/scoring.rs). Callers
/// emit both directions.
fn hawk_line_of_sight_relation_edges(tokens: &[Value]) -> Vec<(i32, i32, i32)> {
    const HAWK_LOS_BETWEEN_0: i32 = 9;
    const HAWK_LOS_BETWEEN_1: i32 = 10;
    const HAWK_LOS_BETWEEN_2: i32 = 11;
    const HAWK_LOS_BETWEEN_3_PLUS: i32 = 12;

    // Per-owner: placed hawks as (coord, node index) and the full wildlife map
    // used for between-species counting.
    let mut hawks_by_owner: HashMap<i64, Vec<(Coord, i32)>> = HashMap::new();
    let mut wildlife_by_owner: HashMap<i64, HashMap<Coord, i64>> = HashMap::new();
    for token in tokens {
        if field(token, "token_kind").and_then(Value::as_str) != Some("placed_tile") {
            continue;
        }
        let Some(coord) = coord_key(field(token, "coord_ref")) else {
            continue;
        };
        let species = safe_i64(field(token, "placed_wildlife"), -1);
        if !(0..WILDLIFE_COUNT as i64).contains(&species) {
            continue;
        }
        let owner = safe_i64(field(token, "owner_seat"), -1);
        wildlife_by_owner
            .entry(owner)
            .or_default()
            .insert(coord, species);
        if species == 3 {
            let node = safe_i64(field(token, "token_index"), -1) as i32;
            hawks_by_owner.entry(owner).or_default().push((coord, node));
        }
    }

    let mut edges = Vec::new();
    for (owner, hawks) in &hawks_by_owner {
        if hawks.len() < 2 {
            continue;
        }
        let hawk_coords: HashSet<Coord> = hawks.iter().map(|(coord, _)| *coord).collect();
        let wildlife = wildlife_by_owner.get(owner);
        for (index, (coord, node)) in hawks.iter().enumerate() {
            // Exclude self so the scan reports partners, matching the engine.
            let mut others = hawk_coords.clone();
            others.remove(coord);
            let empty = HashMap::new();
            let all_wildlife = wildlife.unwrap_or(&empty);
            let partners =
                hawk_los_partner_nodes(*coord, &others, all_wildlife, hawks, index);
            for (target_node, between) in partners {
                let relation_id = match between {
                    0 => HAWK_LOS_BETWEEN_0,
                    1 => HAWK_LOS_BETWEEN_1,
                    2 => HAWK_LOS_BETWEEN_2,
                    _ => HAWK_LOS_BETWEEN_3_PLUS,
                };
                edges.push((*node, target_node, relation_id));
            }
        }
    }
    edges
}

/// Line-of-sight partners of the hawk at `coord` returned as
/// `(partner_node, distinct_species_between)`, deduped so each unordered pair
/// is emitted once (only when the partner's list index exceeds `self_index`).
fn hawk_los_partner_nodes(
    coord: Coord,
    others: &HashSet<Coord>,
    all_wildlife: &HashMap<Coord, i64>,
    hawks: &[(Coord, i32)],
    self_index: usize,
) -> Vec<(i32, u32)> {
    let mut partners = Vec::new();
    for direction in DIRECTIONS {
        let mut current = coord.neighbor(direction);
        let mut distance = 1;
        let mut between: u8 = 0;
        while distance <= MAX_LOS_STEPS {
            if others.contains(&current) {
                if distance > 1
                    && let Some(partner_index) =
                        hawks.iter().position(|(hawk, _)| *hawk == current)
                    && self_index < partner_index
                {
                    partners.push((hawks[partner_index].1, u32::from(between).count_ones()));
                }
                break;
            }
            if let Some(species) = all_wildlife.get(&current).copied()
                && (0..WILDLIFE_COUNT as i64).contains(&species)
                && species != 3
            {
                between |= 1 << species as u8;
            }
            current = current.neighbor(direction);
            distance += 1;
        }
    }
    partners
}

/// Action-source hawk line-of-sight relation edges (Hawk C/D only).
///
/// For each hawk-placing legal action, emits an edge from the action row to
/// every EXISTING placed hawk on the active seat's board that the candidate
/// placement would sit in line of sight of, bucketed by the count of DISTINCT
/// non-hawk species strictly between the pair (mirroring the token->token
/// buckets in `hawk_line_of_sight_relation_edges`, one tier up: ids 13..=16).
/// Unlike ids 9..=12 (token-source, inert for CascadiaFormer because its gated
/// action bias slices those rows away), these live on the action-source rows the
/// model actually consumes.
///
/// This is the SINGLE source of truth for action-sourced hawk LOS geometry: it
/// reuses the same `hawk_los_partner_nodes` scan the semantic Hawk-D dims feed
/// (via `hawk_los_between_counts`), so training (`combined_relation_edges`) and
/// serving (`action_relation_tail`) compute byte-identical edges. Under Hawk A
/// the vector is empty, keeping the AAAAA path byte-identical.
///
/// Returns `(action_index, partner_token_index, relation_id)` where
/// `action_index` is the 0-based index into `legal_actions` (the action's row
/// within the action block); callers add `token_count` to reach its node index.
fn hawk_los_action_edges(root: &Value, active_seat: i64, cards: ScoringCards) -> Vec<(usize, i32, i32)> {
    const ACTION_HAWK_LOS_0: i32 = 13;
    const ACTION_HAWK_LOS_1: i32 = 14;
    const ACTION_HAWK_LOS_2: i32 = 15;
    const ACTION_HAWK_LOS_3_PLUS: i32 = 16;

    // Action-source LOS edges exist only under Hawk C/D; Hawk A stays exactly on
    // the pre-existing ids-5..8 edge set.
    if !matches!(cards.hawk, ScoringVariant::C | ScoringVariant::D) {
        return Vec::new();
    }
    let Some(tokens) = field(root, "public_tokens")
        .and_then(|public_tokens| field(public_tokens, "tokens"))
        .and_then(Value::as_array)
    else {
        return Vec::new();
    };

    // Existing placed hawks and the full wildlife map for the ACTIVE board only:
    // a hawk only ever sees hawks on its own board, and the placing action is the
    // active seat's own move.
    let mut existing_hawks: Vec<(Coord, i32)> = Vec::new();
    let mut active_wildlife: HashMap<Coord, i64> = HashMap::new();
    for token in tokens {
        if field(token, "token_kind").and_then(Value::as_str) != Some("placed_tile") {
            continue;
        }
        if safe_i64(field(token, "owner_seat"), -1) != active_seat {
            continue;
        }
        let Some(coord) = coord_key(field(token, "coord_ref")) else {
            continue;
        };
        let species = safe_i64(field(token, "placed_wildlife"), -1);
        if !(0..WILDLIFE_COUNT as i64).contains(&species) {
            continue;
        }
        active_wildlife.insert(coord, species);
        if species == 3 {
            let node = safe_i64(field(token, "token_index"), -1) as i32;
            existing_hawks.push((coord, node));
        }
    }
    if existing_hawks.is_empty() {
        return Vec::new();
    }
    let existing_coords: HashSet<Coord> = existing_hawks.iter().map(|(coord, _)| *coord).collect();

    let Some(actions) = field(root, "legal_actions").and_then(Value::as_array) else {
        return Vec::new();
    };

    let mut out = Vec::new();
    for (action_index, action) in actions.iter().enumerate() {
        if species_from_action(action) != 3 {
            continue;
        }
        if !bool_field(action, "wildlife_placement_present") {
            continue;
        }
        let Some(candidate) = coord_key(field(action, "wildlife_coord_ref")) else {
            continue;
        };
        // Reuse the token->token scan as the single geometry source: place the
        // candidate hawk at scan index 0 (dummy node) and every existing hawk
        // after it, so `self_index = 0 < partner_index` never dedups a real
        // partner away, and each returned node is an existing hawk token index.
        let mut scan_hawks: Vec<(Coord, i32)> = Vec::with_capacity(existing_hawks.len() + 1);
        scan_hawks.push((candidate, -1));
        scan_hawks.extend(existing_hawks.iter().copied());
        for (partner_node, between) in
            hawk_los_partner_nodes(candidate, &existing_coords, &active_wildlife, &scan_hawks, 0)
        {
            let relation_id = match between {
                0 => ACTION_HAWK_LOS_0,
                1 => ACTION_HAWK_LOS_1,
                2 => ACTION_HAWK_LOS_2,
                _ => ACTION_HAWK_LOS_3_PLUS,
            };
            out.push((action_index, partner_node, relation_id));
        }
    }
    out
}

fn coord_features(coord: Option<&Value>) -> [f32; 6] {
    let Some(coord) = coord else {
        return [0.0; 6];
    };
    [
        normalizer(safe_f64(field(coord, "q"), 0.0), 6.0),
        normalizer(safe_f64(field(coord, "r"), 0.0), 6.0),
        normalizer(safe_f64(field(coord, "s"), 0.0), 6.0),
        if field(coord, "kind").and_then(Value::as_str) == Some("canonical") {
            1.0
        } else {
            0.0
        },
        if field(coord, "kind").and_then(Value::as_str) == Some("overflow") {
            1.0
        } else {
            0.0
        },
        normalizer(
            if field(coord, "cell_index").is_some_and(|value| !value.is_null()) {
                safe_f64(field(coord, "cell_index"), -1.0)
            } else {
                -1.0
            },
            126.0,
        ),
    ]
}

fn relation_degrees(root: &Value) -> Result<HashMap<i64, DegreeSummary>> {
    let mut degrees: HashMap<i64, DegreeSummary> = HashMap::new();
    let relations = field(root, "public_tokens")
        .and_then(|public_tokens| field(public_tokens, "relations"))
        .and_then(Value::as_array)
        .context("root is missing public token relations")?;
    for relation in relations {
        let source = safe_i64(field(relation, "source"), 0);
        let target = safe_i64(field(relation, "target"), 0);
        let kind = field(relation, "relation_kind").and_then(Value::as_str);
        match kind {
            Some("adjacent_hex") => {
                degrees.entry(source).or_default().adjacent_out += 1.0;
                degrees.entry(target).or_default().adjacent_in += 1.0;
                if bool_field(relation, "terrain_matches") {
                    degrees.entry(source).or_default().terrain_match_out += 1.0;
                }
            }
            Some("same_market_slot") => {
                degrees.entry(source).or_default().market_pair += 1.0;
                degrees.entry(target).or_default().market_pair += 1.0;
            }
            _ => {}
        }
    }
    Ok(degrees)
}

pub fn public_token_features(root: &Value) -> Result<Vec<Vec<f32>>> {
    let public_tokens = field(root, "public_tokens").context("root is missing public_tokens")?;
    let degrees = relation_degrees(root)?;
    let tokens = field(public_tokens, "tokens")
        .and_then(Value::as_array)
        .context("public_tokens.tokens missing")?;
    let mut rows = Vec::with_capacity(tokens.len());
    for token in tokens {
        let kind = field(token, "token_kind")
            .and_then(Value::as_str)
            .unwrap_or("");
        let mut row = Vec::with_capacity(PUBLIC_TOKEN_FEATURE_DIM);
        row.extend(
            TOKEN_KINDS
                .iter()
                .map(|expected| if kind == *expected { 1.0 } else { 0.0 }),
        );
        row.extend([
            normalizer(safe_f64(field(token, "owner_seat"), -1.0), 3.0),
            normalizer(safe_f64(field(token, "relative_seat"), -1.0), 3.0),
            normalizer(safe_f64(field(token, "market_slot"), -1.0), 3.0),
        ]);
        row.extend(coord_features(field(token, "coord_ref")));
        let token_index = safe_i64(field(token, "token_index"), -1);
        let degree = degrees.get(&token_index).copied().unwrap_or_default();
        row.extend([
            normalizer(safe_f64(field(token, "nature_tokens"), 0.0), 10.0),
            normalizer(safe_f64(field(token, "tile_count"), 0.0), 23.0),
            normalizer(safe_f64(field(token, "current_base_score"), 0.0), 100.0),
            normalizer(safe_f64(field(token, "current_wildlife_total"), 0.0), 80.0),
            normalizer(safe_f64(field(token, "current_habitat_total"), 0.0), 50.0),
            normalizer(safe_f64(field(token, "tile_id"), 0.0), 84.0),
            normalizer(safe_f64(field(token, "terrain_a"), -1.0), 4.0),
            normalizer(safe_f64(field(token, "terrain_b"), -1.0), 4.0),
            normalizer(safe_f64(field(token, "wildlife_mask"), 0.0), 31.0),
            if bool_field(token, "keystone") {
                1.0
            } else {
                0.0
            },
            normalizer(safe_f64(field(token, "rotation"), 0.0), 5.0),
            normalizer(safe_f64(field(token, "placed_wildlife"), -1.0), 4.0),
            normalizer(safe_f64(field(token, "species"), -1.0), 4.0),
            normalizer(safe_f64(field(token, "neighbor_count"), 0.0), 6.0),
            if bool_field(token, "active_frontier") {
                1.0
            } else {
                0.0
            },
            normalizer(degree.adjacent_out, 6.0),
            normalizer(degree.adjacent_in, 6.0),
            normalizer(degree.terrain_match_out, 6.0),
            normalizer(degree.market_pair, 2.0),
        ]);
        let wildlife_bag = field(token, "wildlife_bag").and_then(Value::as_array);
        for index in 0..WILDLIFE_COUNT {
            row.push(normalizer(
                wildlife_bag
                    .and_then(|values| values.get(index))
                    .map_or(0.0, |value| safe_f64(Some(value), 0.0)),
                100.0,
            ));
        }
        let terrain_capacity_sum = field(token, "unseen_tile_terrain_capacity")
            .and_then(Value::as_array)
            .map(|values| values.iter().map(|value| safe_f64(Some(value), 0.0)).sum())
            .unwrap_or(0.0);
        let wildlife_capacity_sum = field(token, "unseen_tile_wildlife_capacity")
            .and_then(Value::as_array)
            .map(|values| values.iter().map(|value| safe_f64(Some(value), 0.0)).sum())
            .unwrap_or(0.0);
        row.extend([
            normalizer(terrain_capacity_sum, 100.0),
            normalizer(wildlife_capacity_sum, 100.0),
        ]);
        if row.len() != PUBLIC_TOKEN_FEATURE_DIM {
            bail!("public token feature dimension mismatch: {}", row.len());
        }
        rows.push(row);
    }
    Ok(rows)
}

fn action_immediate_score(action: &Value) -> f32 {
    safe_f64(field(action, "immediate_pre_rollout_base_score"), 0.0)
}

fn tile_id_from_ref(action: &Value) -> f32 {
    let tile_ref = field(action, "tile_ref")
        .and_then(Value::as_str)
        .unwrap_or("");
    if !tile_ref.starts_with("tile:") {
        return 0.0;
    }
    tile_ref
        .split_once(':')
        .and_then(|(_, rest)| rest.split_once('@').map(|(id, _)| id))
        .and_then(|id| id.parse::<f32>().ok())
        .unwrap_or(0.0)
}

fn species_one_hot(action: &Value) -> [f32; WILDLIFE_COUNT] {
    let wildlife_ref = field(action, "wildlife_ref")
        .and_then(Value::as_str)
        .unwrap_or("");
    let mut out = [0.0; WILDLIFE_COUNT];
    for (index, species) in SPECIES_NAMES.iter().enumerate() {
        if wildlife_ref.starts_with(species) {
            out[index] = 1.0;
        }
    }
    out
}

fn merit_action_features(root: &Value) -> Result<Vec<Vec<f32>>> {
    let actions = field(root, "legal_actions")
        .and_then(Value::as_array)
        .context("root legal_actions missing")?;
    let mut rows = Vec::with_capacity(actions.len());
    for action in actions {
        let mut row = Vec::with_capacity(MERIT_ACTION_FEATURE_DIM);
        row.extend([
            normalizer(safe_f64(field(action, "active_seat"), 0.0), 3.0),
            safe_f64(field(action, "nature_spend"), 0.0),
            normalizer(safe_f64(field(action, "draft_slot"), 0.0), 3.0),
            normalizer(safe_f64(field(action, "rotation"), 0.0), 5.0),
        ]);
        row.extend(coord_features(field(action, "target_coord_ref")));
        row.extend(coord_features(field(action, "wildlife_coord_ref")));
        row.extend([
            normalizer(action_immediate_score(action), 100.0),
            if bool_field(action, "wildlife_placement_present") {
                1.0
            } else {
                0.0
            },
            normalizer(tile_id_from_ref(action), 84.0),
            if field(action, "cleanup_choice").and_then(Value::as_str) == Some("none") {
                0.0
            } else {
                1.0
            },
        ]);
        row.extend(species_one_hot(action));
        if row.len() != MERIT_ACTION_FEATURE_DIM {
            bail!("action feature dimension mismatch: {}", row.len());
        }
        rows.push(row);
    }
    Ok(rows)
}

fn public_token_action_features(root: &Value) -> Result<Vec<Vec<f32>>> {
    let base_rows = merit_action_features(root)?;
    let actions = field(root, "legal_actions")
        .and_then(Value::as_array)
        .context("root legal_actions missing")?;
    let mut rows = Vec::with_capacity(actions.len());
    for (base, action) in base_rows.into_iter().zip(actions.iter()) {
        let mut row = base;
        row.extend([
            normalizer(
                safe_f64(
                    field(action, "tile_slot").or_else(|| field(action, "draft_slot")),
                    0.0,
                ),
                3.0,
            ),
            normalizer(
                safe_f64(
                    field(action, "wildlife_slot").or_else(|| field(action, "draft_slot")),
                    0.0,
                ),
                3.0,
            ),
            normalizer(
                safe_f64(field(action, "tile_id"), tile_id_from_ref(action)),
                84.0,
            ),
            normalizer(safe_f64(field(action, "tile_terrain_a"), -1.0), 4.0),
            normalizer(safe_f64(field(action, "tile_terrain_b"), -1.0), 4.0),
            normalizer(safe_f64(field(action, "tile_wildlife_mask"), 0.0), 31.0),
            if bool_field(action, "tile_keystone") {
                1.0
            } else {
                0.0
            },
            normalizer(safe_f64(field(action, "wildlife_species"), -1.0), 4.0),
        ]);
        if row.len() != PUBLIC_TOKEN_ACTION_FEATURE_DIM {
            bail!(
                "public token action feature dimension mismatch: {}",
                row.len()
            );
        }
        rows.push(row);
    }
    Ok(rows)
}

fn species_from_action(action: &Value) -> i64 {
    safe_i64(field(action, "wildlife_species"), -1)
}

fn wildlife_mask_contains(mask: Option<&Value>, species: i64) -> bool {
    if species < 0 || species >= WILDLIFE_COUNT as i64 {
        return false;
    }
    let mask = safe_i64(mask, 0);
    (mask & (1_i64 << species)) != 0
}

fn tile_terrain_on_edge(tile: TileInfo, edge: i32) -> i32 {
    if tile.terrain_b < 0 {
        return tile.terrain_a;
    }
    let rotation = tile.rotation.rem_euclid(6);
    let offset = (edge + 6 - rotation).rem_euclid(6);
    if offset < 3 {
        tile.terrain_a
    } else {
        tile.terrain_b
    }
}

fn state_view(root: &Value) -> Result<StateView> {
    let active_seat = safe_i64(field(root, "active_seat"), 0);
    let mut state = StateView {
        active_seat,
        ..StateView::default()
    };
    let tokens = field(root, "public_tokens")
        .and_then(|public_tokens| field(public_tokens, "tokens"))
        .and_then(Value::as_array)
        .context("root public_tokens.tokens missing")?;
    for token in tokens {
        let kind = field(token, "token_kind")
            .and_then(Value::as_str)
            .unwrap_or("");
        match kind {
            "placed_tile" => {
                let owner = safe_i64(field(token, "owner_seat"), -1);
                let Some(coord) = coord_key(field(token, "coord_ref")) else {
                    continue;
                };
                let tile = TileInfo {
                    terrain_a: safe_i64(field(token, "terrain_a"), -1) as i32,
                    terrain_b: safe_i64(field(token, "terrain_b"), -1) as i32,
                    rotation: safe_i64(field(token, "rotation"), 0) as i32,
                };
                if owner == active_seat {
                    state.active_tiles.insert(coord, tile);
                }
                let wildlife = safe_i64(field(token, "placed_wildlife"), -1);
                if (0..WILDLIFE_COUNT as i64).contains(&wildlife) {
                    state
                        .wildlife_by_owner
                        .entry(owner)
                        .or_default()
                        .entry(wildlife)
                        .or_default()
                        .insert(coord);
                    if owner == active_seat {
                        state
                            .active_wildlife
                            .entry(wildlife)
                            .or_default()
                            .insert(coord);
                    }
                } else if owner == active_seat {
                    let mask = safe_i64(field(token, "wildlife_mask"), 0);
                    for species in 0..WILDLIFE_COUNT {
                        if (mask & (1_i64 << species)) != 0 {
                            state.empty_species_slots[species] += 1;
                        }
                    }
                }
            }
            "market_wildlife" => {
                let species = safe_i64(field(token, "species"), -1);
                if (0..WILDLIFE_COUNT as i64).contains(&species) {
                    state.market_species_counts[species as usize] += 1;
                }
            }
            "public_supply" => {
                if let Some(values) = field(token, "wildlife_bag").and_then(Value::as_array) {
                    for species in 0..WILDLIFE_COUNT.min(values.len()) {
                        state.supply_bag[species] = safe_f64(values.get(species), 0.0);
                    }
                }
                if let Some(values) =
                    field(token, "unseen_tile_wildlife_capacity").and_then(Value::as_array)
                {
                    for species in 0..WILDLIFE_COUNT.min(values.len()) {
                        state.supply_capacity[species] = safe_f64(values.get(species), 0.0);
                    }
                }
            }
            _ => {}
        }
    }
    Ok(state)
}

fn line_length_through(coord: Coord, positions: &HashSet<Coord>, direction: (i32, i32)) -> i32 {
    let mut length = 1;
    let mut current = coord.neighbor(direction);
    while positions.contains(&current) {
        length += 1;
        current = current.neighbor(direction);
    }
    let opposite = (-direction.0, -direction.1);
    current = coord.neighbor(opposite);
    while positions.contains(&current) {
        length += 1;
        current = current.neighbor(opposite);
    }
    length
}

fn component_size(coord: Coord, positions: &HashSet<Coord>) -> i32 {
    if !positions.contains(&coord) {
        return 0;
    }
    let mut seen = HashSet::from([coord]);
    let mut stack = vec![coord];
    while let Some(current) = stack.pop() {
        for direction in DIRECTIONS {
            let neighbor = current.neighbor(direction);
            if positions.contains(&neighbor) && seen.insert(neighbor) {
                stack.push(neighbor);
            }
        }
    }
    seen.len() as i32
}

fn hawk_line_of_sight_count(coord: Coord, hawks: &HashSet<Coord>) -> i32 {
    let mut count = 0;
    for direction in DIRECTIONS {
        let mut current = coord.neighbor(direction);
        let mut distance = 1;
        while distance <= 16 {
            if hawks.contains(&current) {
                if distance > 1 {
                    count += 1;
                }
                break;
            }
            current = current.neighbor(direction);
            distance += 1;
        }
    }
    count
}

/// Upper bound on line-of-sight step length. No Cascadia board straight line
/// spans the canonical grid, so bounding by its diameter mirrors the engine's
/// `to_index().is_none()` stop condition while guaranteeing termination.
const MAX_LOS_STEPS: i32 = GRID_DIM as i32;

fn coords_adjacent(a: Coord, b: Coord) -> bool {
    DIRECTIONS.iter().any(|direction| a.neighbor(*direction) == b)
}

/// Line-of-sight partners of the hawk at `coord`, each reported as the count of
/// DISTINCT non-hawk species strictly between the two hawks. Mirrors the engine
/// `hawk_lines_of_sight` scan (see `scoring.rs`): the first hawk encountered in
/// a direction is the partner; intervening non-hawk wildlife only contributes
/// to the between-count. `coord` itself must be excluded from `hawks`.
fn hawk_los_between_counts(
    coord: Coord,
    hawks: &HashSet<Coord>,
    all_wildlife: &HashMap<Coord, i64>,
) -> Vec<u32> {
    let mut partners = Vec::new();
    for direction in DIRECTIONS {
        let mut current = coord.neighbor(direction);
        let mut distance = 1;
        let mut between: u8 = 0;
        while distance <= MAX_LOS_STEPS {
            if hawks.contains(&current) {
                if distance > 1 {
                    partners.push(u32::from(between).count_ones());
                }
                break;
            }
            if let Some(species) = all_wildlife.get(&current).copied()
                && (0..WILDLIFE_COUNT as i64).contains(&species)
                && species != 3
            {
                between |= 1 << species as u8;
            }
            current = current.neighbor(direction);
            distance += 1;
        }
    }
    partners
}

/// Sizes of every connected component in `positions`. JSON-space mirror of the
/// engine's `wildlife_components` size list (see `bear_component_sizes` in
/// crates/cascadia-game/scoring.rs), used for Bear-C set bookkeeping.
fn all_component_sizes(positions: &HashSet<Coord>) -> Vec<usize> {
    let mut remaining = positions.clone();
    let mut sizes = Vec::new();
    while let Some(start) = remaining.iter().next().copied() {
        remaining.remove(&start);
        let mut size = 1usize;
        let mut stack = vec![start];
        while let Some(current) = stack.pop() {
            for direction in DIRECTIONS {
                let neighbor = current.neighbor(direction);
                if remaining.remove(&neighbor) {
                    size += 1;
                    stack.push(neighbor);
                }
            }
        }
        sizes.push(size);
    }
    sizes
}

/// Bear-C set progress: how many of the three distinct group sizes {1, 2, 3}
/// are currently represented among `sizes` (0..=3). Holding one group of each
/// is what earns the +3 set bonus in `score_bears` (`ScoringVariant::C`).
fn bear_set_progress(sizes: &[usize]) -> u32 {
    let mut present = [false; 3];
    for &size in sizes {
        if (1..=3).contains(&size) {
            present[size - 1] = true;
        }
    }
    present.iter().filter(|seen| **seen).count() as u32
}

/// Compactness tier of the elk shape the placement at `coord` joins, matching
/// the Elk-B / connected-shape ladder: 0 isolated, 1 in a line/pair, 2 in a
/// triangle, 3 in a triangle-plus-one fan.
fn elk_shape_level(coord: Coord, positions: &HashSet<Coord>) -> i32 {
    let elk_neighbors: Vec<Coord> = DIRECTIONS
        .iter()
        .map(|direction| coord.neighbor(*direction))
        .filter(|neighbor| positions.contains(neighbor))
        .collect();
    if elk_neighbors.is_empty() {
        return 0;
    }
    let mut level = 1;
    for i in 0..elk_neighbors.len() {
        for j in (i + 1)..elk_neighbors.len() {
            if !coords_adjacent(elk_neighbors[i], elk_neighbors[j]) {
                continue;
            }
            level = level.max(2);
            let triangle = [coord, elk_neighbors[i], elk_neighbors[j]];
            for candidate in positions {
                if triangle.contains(candidate) {
                    continue;
                }
                let shared = triangle
                    .iter()
                    .filter(|corner| coords_adjacent(**corner, *candidate))
                    .count();
                if shared >= 2 {
                    level = level.max(3);
                }
            }
        }
    }
    level
}

fn habitat_edge_counts(action: &Value, state: &StateView) -> (i32, i32, i32) {
    let Some(target) = coord_key(field(action, "target_coord_ref")) else {
        return (0, 0, 6);
    };
    let mut matches = 0;
    let mut mismatches = 0;
    let action_tile = TileInfo {
        terrain_a: safe_i64(field(action, "tile_terrain_a"), -1) as i32,
        terrain_b: safe_i64(field(action, "tile_terrain_b"), -1) as i32,
        rotation: safe_i64(field(action, "rotation"), 0) as i32,
    };
    for (edge, direction) in DIRECTIONS.iter().enumerate() {
        let neighbor_coord = target.neighbor(*direction);
        let Some(neighbor_tile) = state.active_tiles.get(&neighbor_coord).copied() else {
            continue;
        };
        let action_terrain = tile_terrain_on_edge(action_tile, edge as i32);
        let neighbor_terrain = tile_terrain_on_edge(neighbor_tile, ((edge + 3) % 6) as i32);
        if action_terrain == neighbor_terrain {
            matches += 1;
        } else {
            mismatches += 1;
        }
    }
    (matches, mismatches, 6 - matches - mismatches)
}

fn semantic_action_features(root: &Value, cards: ScoringCards) -> Result<Vec<Vec<f32>>> {
    let state = state_view(root)?;
    let actions = field(root, "legal_actions")
        .and_then(Value::as_array)
        .context("root legal_actions missing")?;
    let all_active_wildlife: HashMap<Coord, i64> = state
        .active_wildlife
        .iter()
        .flat_map(|(species, coords)| coords.iter().map(|coord| (*coord, *species)))
        .collect();
    let mut rows = Vec::with_capacity(actions.len());
    for action in actions {
        let species = species_from_action(action);
        let wildlife_coord = coord_key(field(action, "wildlife_coord_ref"));
        let wildlife_present =
            bool_field(action, "wildlife_placement_present") && wildlife_coord.is_some();
        let (matches, mismatches, open_edges) = habitat_edge_counts(action, &state);
        let target_neighbor_count = matches + mismatches;
        let active_species_before = if (0..WILDLIFE_COUNT as i64).contains(&species) {
            state.active_wildlife.get(&species).map_or(0, HashSet::len)
        } else {
            0
        };
        let mut opponent_max = 0usize;
        for (owner, by_species) in &state.wildlife_by_owner {
            if *owner == state.active_seat {
                continue;
            }
            opponent_max = opponent_max.max(by_species.get(&species).map_or(0, HashSet::len));
        }
        let mut after_species_positions = state.active_wildlife.clone();
        if (0..WILDLIFE_COUNT as i64).contains(&species) && wildlife_present {
            after_species_positions
                .entry(species)
                .or_default()
                .insert(wildlife_coord.expect("checked above"));
        }
        let mut after_all_wildlife = all_active_wildlife.clone();
        if (0..WILDLIFE_COUNT as i64).contains(&species) && wildlife_present {
            after_all_wildlife.insert(wildlife_coord.expect("checked above"), species);
        }
        let mut same_neighbors = 0;
        let mut any_neighbors = 0;
        let mut other_species = HashSet::new();
        if let Some(coord) = wildlife_coord {
            for direction in DIRECTIONS {
                if let Some(neighbor_species) =
                    after_all_wildlife.get(&coord.neighbor(direction)).copied()
                {
                    any_neighbors += 1;
                    if neighbor_species == species {
                        same_neighbors += 1;
                    } else {
                        other_species.insert(neighbor_species);
                    }
                }
            }
        }
        let bear_pair_signal = if species == 0 && same_neighbors == 1 {
            1.0
        } else {
            0.0
        };
        let bear_overcluster_signal = if species == 0 && same_neighbors > 1 {
            1.0
        } else {
            0.0
        };

        let mut elk_line_length = 0;
        if species == 1
            && let Some(coord) = wildlife_coord
            && let Some(elk_positions) = after_species_positions.get(&1)
            && elk_positions.contains(&coord)
        {
            elk_line_length = AXIS_DIRECTIONS
                .iter()
                .map(|direction| line_length_through(coord, elk_positions, *direction))
                .max()
                .unwrap_or(0);
        }

        let mut salmon_component_size = 0;
        let mut salmon_degree = 0;
        let mut salmon_branch_risk = 0;
        if species == 2
            && let Some(coord) = wildlife_coord
            && let Some(salmon_positions) = after_species_positions.get(&2)
            && salmon_positions.contains(&coord)
        {
            salmon_component_size = component_size(coord, salmon_positions);
            salmon_degree = DIRECTIONS
                .iter()
                .filter(|direction| salmon_positions.contains(&coord.neighbor(**direction)))
                .count() as i32;
            let neighbor_has_branch = DIRECTIONS.iter().any(|direction| {
                let neighbor = coord.neighbor(*direction);
                salmon_positions.contains(&neighbor)
                    && DIRECTIONS
                        .iter()
                        .filter(|second| salmon_positions.contains(&neighbor.neighbor(**second)))
                        .count()
                        > 2
            });
            salmon_branch_risk = if salmon_degree > 2 || neighbor_has_branch {
                1
            } else {
                0
            };
        }

        let hawks = after_species_positions.get(&3).cloned().unwrap_or_default();
        let hawk_isolated = if species == 3 && same_neighbors == 0 {
            1.0
        } else {
            0.0
        };
        let hawk_los = if species == 3 {
            wildlife_coord
                .map(|coord| hawk_line_of_sight_count(coord, &hawks))
                .unwrap_or(0)
        } else {
            0
        };
        let hawk_adjacent_penalty = if species == 3 && same_neighbors > 0 {
            1.0
        } else {
            0.0
        };

        let mut fox_unique = 0usize;
        let mut fox_nonfox = 0usize;
        if species == 4
            && let Some(coord) = wildlife_coord
        {
            let adjacent_species = DIRECTIONS
                .iter()
                .filter_map(|direction| {
                    after_all_wildlife.get(&coord.neighbor(*direction)).copied()
                })
                .collect::<Vec<_>>();
            fox_unique = adjacent_species
                .iter()
                .copied()
                .collect::<HashSet<_>>()
                .len();
            fox_nonfox = adjacent_species
                .iter()
                .filter(|neighbor_species| **neighbor_species != 4)
                .count();
        }

        // --- Card-aware semantic hints ---------------------------------
        // Six dims are Card-A-shaped. Under Card A each reproduces the exact
        // legacy value (byte-identical AAAAA output is a hard invariant); under
        // the CBDDB variant for that species it instead carries a hint matched
        // to how that card actually scores (see crates/cascadia-game/scoring.rs).
        let bear_dim_pair;
        let bear_dim_over;
        match cards.bear {
            ScoringVariant::A => {
                bear_dim_pair = bear_pair_signal;
                bear_dim_over = bear_overcluster_signal;
            }
            _ => {
                // Bear C's defining structure is the +3 bonus for holding one
                // group each of sizes {1,2,3}. The model must plan toward
                // completing that set, so encode:
                //   dim 1 = current board set progress (0..3 distinct sizes
                //           held) BEFORE this action, broadcast to every action
                //           row as shared context;
                //   dim 2 = this placement's MARGINAL effect on set progress,
                //           centered at 0.5 (neutral): >0.5 advances a
                //           still-missing size, <0.5 regresses (e.g. merging a
                //           size-1 and size-2 into a size-3 drops two held
                //           sizes) or overgrows past 3.
                let before_progress = state
                    .active_wildlife
                    .get(&0)
                    .map(|bears| bear_set_progress(&all_component_sizes(bears)))
                    .unwrap_or(0);
                let after_progress = if species == 0
                    && wildlife_present
                    && let Some(bears) = after_species_positions.get(&0)
                {
                    bear_set_progress(&all_component_sizes(bears))
                } else {
                    before_progress
                };
                let delta = after_progress as i32 - before_progress as i32;
                bear_dim_pair = normalizer(before_progress as f32, 3.0);
                bear_dim_over = (0.5 + delta as f32 / 2.0).clamp(0.0, 1.0);
            }
        }

        let elk_dim = match cards.elk {
            ScoringVariant::A => normalizer(elk_line_length.min(4) as f32, 4.0),
            _ => {
                // Elk B rewards compact connected shapes (pair/triangle/fan)
                // instead of straight-line length.
                let level = if species == 1
                    && let Some(coord) = wildlife_coord
                    && let Some(elk) = after_species_positions.get(&1)
                    && elk.contains(&coord)
                {
                    elk_shape_level(coord, elk)
                } else {
                    0
                };
                normalizer(level as f32, 3.0)
            }
        };

        let hawk_dim_iso;
        let hawk_dim_pen;
        match cards.hawk {
            ScoringVariant::A => {
                hawk_dim_iso = hawk_isolated;
                hawk_dim_pen = hawk_adjacent_penalty;
            }
            _ => {
                // Hawk D scores a max-weight matching over LINE-OF-SIGHT pairs
                // weighted by DISTINCT species between (0/4/7/9), so adjacency
                // isolation/penalty is meaningless. Surface the best reachable
                // pair weight and the number of "productive" (>=1 species
                // between) partners for the placed hawk. The plain LOS partner
                // count stays in its own general dim below.
                let (best_weight, productive) = if species == 3
                    && let Some(coord) = wildlife_coord
                {
                    let counts = hawk_los_between_counts(coord, &hawks, &after_all_wildlife);
                    let best = counts
                        .iter()
                        .map(|between| match between {
                            0 => 0u16,
                            1 => 4,
                            2 => 7,
                            _ => 9,
                        })
                        .max()
                        .unwrap_or(0);
                    let productive = counts.iter().filter(|between| **between >= 1).count();
                    (best, productive)
                } else {
                    (0, 0)
                };
                hawk_dim_iso = normalizer(best_weight as f32, 9.0);
                hawk_dim_pen = normalizer(productive.min(6) as f32, 6.0);
            }
        }

        let fox_dim = match cards.fox {
            ScoringVariant::A => normalizer(fox_unique as f32, 5.0),
            _ => {
                // Fox B scores each fox by how many neighbor species appear as
                // a PAIR (>=2 adjacent) — 0/3/5/7 for 0/1/2/3+ — not by the
                // count of distinct neighbor species.
                let pair_types = if species == 4
                    && let Some(coord) = wildlife_coord
                {
                    let mut counts = [0i32; WILDLIFE_COUNT];
                    for direction in DIRECTIONS {
                        if let Some(neighbor_species) =
                            after_all_wildlife.get(&coord.neighbor(direction)).copied()
                            && (0..WILDLIFE_COUNT as i64).contains(&neighbor_species)
                            && neighbor_species != 4
                        {
                            counts[neighbor_species as usize] += 1;
                        }
                    }
                    counts.iter().filter(|count| **count >= 2).count()
                } else {
                    0
                };
                normalizer(pair_types as f32, 3.0)
            }
        };

        let supply_bag = if (0..WILDLIFE_COUNT as i64).contains(&species) {
            state.supply_bag[species as usize]
        } else {
            0.0
        };
        let supply_capacity = if (0..WILDLIFE_COUNT as i64).contains(&species) {
            state.supply_capacity[species as usize]
        } else {
            0.0
        };
        let empty_species_slots = if (0..WILDLIFE_COUNT as i64).contains(&species) {
            state.empty_species_slots[species as usize]
        } else {
            0
        };
        let market_species_counts = if (0..WILDLIFE_COUNT as i64).contains(&species) {
            state.market_species_counts[species as usize]
        } else {
            0
        };
        let semantic = vec![
            normalizer(target_neighbor_count as f32, 6.0),
            normalizer(matches as f32, 6.0),
            normalizer(mismatches as f32, 6.0),
            normalizer(open_edges as f32, 6.0),
            if wildlife_mask_contains(field(action, "tile_wildlife_mask"), species) {
                1.0
            } else {
                0.0
            },
            normalizer(
                safe_i64(field(action, "tile_wildlife_mask"), 0).count_ones() as f32,
                5.0,
            ),
            normalizer(active_species_before as f32, 20.0),
            normalizer(empty_species_slots as f32, 20.0),
            normalizer(market_species_counts as f32, 4.0),
            normalizer(opponent_max as f32, 20.0),
            normalizer(opponent_max as f32 - active_species_before as f32, 20.0),
            normalizer(same_neighbors as f32, 6.0),
            normalizer(any_neighbors as f32, 6.0),
            normalizer(other_species.len() as f32, 4.0),
            bear_dim_pair,
            bear_dim_over,
            elk_dim,
            normalizer(
                if species == 1 {
                    same_neighbors as f32
                } else {
                    0.0
                },
                6.0,
            ),
            normalizer(salmon_component_size.min(7) as f32, 7.0),
            normalizer(salmon_degree.min(3) as f32, 3.0),
            salmon_branch_risk as f32,
            hawk_dim_iso,
            normalizer(hawk_los.min(6) as f32, 6.0),
            hawk_dim_pen,
            fox_dim,
            normalizer(fox_nonfox as f32, 6.0),
            normalizer(supply_bag, 100.0),
            normalizer(supply_capacity, 100.0),
        ];
        if semantic.len() != SEMANTIC_ACTION_FEATURE_DIM {
            bail!("semantic feature dimension mismatch: {}", semantic.len());
        }
        rows.push(semantic);
    }
    Ok(rows)
}

pub fn semantic_public_token_action_features(
    root: &Value,
    cards: ScoringCards,
) -> Result<Vec<Vec<f32>>> {
    let base_rows = public_token_action_features(root)?;
    let semantic_rows = semantic_action_features(root, cards)?;
    let mut rows = Vec::with_capacity(base_rows.len());
    for (mut base, semantic) in base_rows.into_iter().zip(semantic_rows.into_iter()) {
        base.extend(semantic);
        if base.len() != SEMANTIC_PUBLIC_TOKEN_ACTION_FEATURE_DIM {
            bail!(
                "semantic public-token action feature dimension mismatch: {}",
                base.len()
            );
        }
        rows.push(base);
    }
    Ok(rows)
}

pub fn selected_action_index(record: &Value) -> Result<usize> {
    let selected = field(record, "selected_action")
        .and_then(Value::as_str)
        .context("record selected_action missing")?;
    let actions = field(record, "legal_actions")
        .and_then(Value::as_array)
        .context("record legal_actions missing")?;
    actions
        .iter()
        .position(|action| field(action, "action_id").and_then(Value::as_str) == Some(selected))
        .with_context(|| {
            format!(
                "selected action missing from legal actions for {:?}",
                string_field(record, "state_hash")
            )
        })
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{
        ScoringCards, ScoringVariant, action_relation_tail, action_relation_tail_reference,
        coord_features, combined_relation_edges, normalizer, selected_action_index,
        semantic_action_features,
    };

    fn placed(index: i64, owner: i64, q: i64, r: i64, wildlife: Option<i64>) -> serde_json::Value {
        let mut token = json!({
            "token_index": index,
            "token_kind": "placed_tile",
            "owner_seat": owner,
            "coord_ref": {"kind": "canonical", "q": q, "r": r},
            "terrain_a": 0,
            "terrain_b": -1,
            "rotation": 0,
            "wildlife_mask": 0,
        });
        if let Some(species) = wildlife {
            token["placed_wildlife"] = json!(species);
        }
        token
    }

    fn place_wildlife_action(species: i64, q: i64, r: i64) -> serde_json::Value {
        json!({
            "action_id": "place",
            "wildlife_species": species,
            "wildlife_placement_present": true,
            "wildlife_coord_ref": {"kind": "canonical", "q": q, "r": r},
        })
    }

    fn root(active_seat: i64, tokens: Vec<serde_json::Value>, actions: Vec<serde_json::Value>) -> serde_json::Value {
        json!({
            "active_seat": active_seat,
            "public_tokens": {"tokens": tokens, "relations": []},
            "legal_actions": actions,
        })
    }

    fn cbddb_with(species: char, variant: ScoringVariant) -> ScoringCards {
        let mut cards = ScoringCards::AAAAA;
        match species {
            'b' => cards.bear = variant,
            'e' => cards.elk = variant,
            'h' => cards.hawk = variant,
            'f' => cards.fox = variant,
            _ => unreachable!(),
        }
        cards
    }

    fn find_edge(edges: &[[i32; 3]], source: i32, target: i32) -> Option<i32> {
        edges
            .iter()
            .find(|edge| edge[0] == source && edge[1] == target)
            .map(|edge| edge[2])
    }

    #[test]
    fn hawk_los_relation_edges_gate_on_hawk_variant() {
        // Active seat hawks at (0,0)#0 and (3,0)#2 with one elk (#1) between:
        // a line-of-sight pair with a single distinct species between -> the
        // bucket-1 relation id (10).
        let tokens = vec![
            placed(0, 0, 0, 0, Some(3)),
            placed(1, 0, 1, 0, Some(1)),
            placed(2, 0, 3, 0, Some(3)),
        ];
        let actions = vec![json!({"action_id": "noop"})];
        let record = root(0, tokens, actions);

        let aaaaa = combined_relation_edges(&record, 3, 1, ScoringCards::AAAAA).unwrap();
        assert!(
            aaaaa.iter().all(|edge| !(9..=12).contains(&edge[2])),
            "AAAAA must never emit a hawk line-of-sight edge: {aaaaa:?}",
        );
        // Same-owner-board edge still connects the two hawk tokens under AAAAA.
        assert_eq!(find_edge(&aaaaa, 0, 2), Some(4));

        let cbddb = combined_relation_edges(&record, 3, 1, ScoringCards::CBDDB).unwrap();
        // Bucketed id 10 (exactly one distinct species between), both
        // directions, overwriting the generic same-owner edge for this pair.
        assert_eq!(find_edge(&cbddb, 0, 2), Some(10));
        assert_eq!(find_edge(&cbddb, 2, 0), Some(10));
        // Non-hawk pairs keep their same-owner-board relation.
        assert_eq!(find_edge(&cbddb, 0, 1), Some(4));
    }

    #[test]
    fn hawk_los_edges_bucket_by_species_between_and_stay_per_board() {
        // Two distinct species (elk + salmon) between the active hawks -> id
        // 11; an opponent hawk pair on a different board must not link.
        let tokens = vec![
            placed(0, 0, 0, 0, Some(3)),
            placed(1, 0, 1, 0, Some(1)),
            placed(2, 0, 2, 0, Some(2)),
            placed(3, 0, 4, 0, Some(3)),
            // Opponent (seat 1) hawks in line of sight on their own board.
            placed(4, 1, 0, 0, Some(3)),
            placed(5, 1, 3, 0, Some(3)),
        ];
        let record = root(0, tokens, vec![json!({"action_id": "noop"})]);
        let cbddb = combined_relation_edges(&record, 6, 1, ScoringCards::CBDDB).unwrap();
        assert_eq!(find_edge(&cbddb, 0, 3), Some(11));
        // Opponent hawks (#4,#5) are on seat 1's board: LOS edge present there
        // too, but never crossing boards.
        assert_eq!(find_edge(&cbddb, 4, 5), Some(9));
        assert_eq!(find_edge(&cbddb, 0, 4), None);
        assert_eq!(find_edge(&cbddb, 0, 5), None);
    }

    #[test]
    fn action_relation_tail_matches_reference_for_hawk_los_action_edges() {
        // Acceptance gate for train/serve parity of action-sourced hawk LOS.
        //
        // Active seat 0 board (all on the q-axis so LOS is a straight scan):
        //   #0 hawk (0,0), #1 elk (1,0), #2 hawk (5,0), #3 salmon (3,0),
        //   #4 elk (4,0). Existing hawks #0 and #2 are already a LOS pair.
        // Candidate hawk placement action at (2,0):
        //   - scanning -q toward (0,0): one distinct species between (elk at
        //     (1,0)) -> bucket 1 -> id 14 on partner token #0;
        //   - scanning +q toward (5,0): two distinct species between (salmon at
        //     (3,0) + elk at (4,0)) -> bucket 2 -> id 15 on partner token #2.
        let tokens = vec![
            placed(0, 0, 0, 0, Some(3)),
            placed(1, 0, 1, 0, Some(1)),
            placed(2, 0, 5, 0, Some(3)),
            placed(3, 0, 3, 0, Some(2)),
            placed(4, 0, 4, 0, Some(1)),
        ];
        let actions = vec![place_wildlife_action(3, 2, 0)];
        let record = root(0, tokens, actions);
        let token_count = 5usize;
        let action_count = 1usize;
        let seq_len = token_count + action_count;

        for cards in [ScoringCards::AAAAA, ScoringCards::CBDDB] {
            // THE gate: the serve fast path must byte-equal the reference
            // (sliced `combined_relation_edges`) for BOTH rulesets.
            let fast = action_relation_tail(&record, token_count, action_count, cards).unwrap();
            let reference =
                action_relation_tail_reference(&record, token_count, action_count, cards).unwrap();
            assert_eq!(
                fast, reference,
                "action_relation_tail must byte-match reference under {cards:?}",
            );
            assert_eq!(fast.len(), action_count * seq_len);

            let row0 = &fast[0..seq_len];
            if matches!(cards.hawk, ScoringVariant::C | ScoringVariant::D) {
                // Non-zero LOS action-edge ids on the hawk-placing action row.
                assert_eq!(row0[0], 14, "partner hawk #0 -> bucket-1 id 14 under {cards:?}");
                assert_eq!(row0[2], 15, "partner hawk #2 -> bucket-2 id 15 under {cards:?}");
                assert!(
                    fast.iter().any(|value| (13..=16).contains(value)),
                    "CBDDB tail must carry action-sourced LOS ids 13..=16",
                );
            } else {
                // AAAAA: no action-sourced LOS edges; every id stays <= 8.
                assert_eq!(row0[0], 0, "no LOS edge to #0 under {cards:?}");
                assert_eq!(row0[2], 0, "no LOS edge to #2 under {cards:?}");
                assert!(
                    fast.iter().all(|value| *value <= 8),
                    "AAAAA tail must never carry a LOS action-edge id (>8)",
                );
            }
        }
    }

    #[test]
    fn bear_dims_encode_set_completion_under_bear_c() {
        // Active board: a lone bear (#0, size 1) plus two separate size-2
        // groups (#1/#2 and #3/#4). Distinct sizes held = {1, 2} -> progress 2.
        let tokens = vec![
            placed(0, 0, 0, 0, Some(0)),
            placed(1, 0, 5, 0, Some(0)),
            placed(2, 0, 6, 0, Some(0)),
            placed(3, 0, 5, 3, Some(0)),
            placed(4, 0, 6, 3, Some(0)),
        ];
        // Placement grows the second size-2 group into a size-3 group,
        // completing the {1,2,3} set: progress 2 -> 3 (advance).
        let actions = vec![place_wildlife_action(0, 7, 3)];
        let record = root(0, tokens, actions);

        let aaaaa = semantic_action_features(&record, ScoringCards::AAAAA).unwrap();
        // Placement at (7,3) touches exactly one bear (6,3): legacy Card-A
        // pair signal fires, overcluster does not.
        assert_eq!(aaaaa[0][14], 1.0);
        assert_eq!(aaaaa[0][15], 0.0);

        let bear_c = semantic_action_features(&record, cbddb_with('b', ScoringVariant::C)).unwrap();
        // dim 1: current set progress 2/3 before the action.
        assert!((bear_c[0][14] - 2.0 / 3.0).abs() < 1e-6, "got {}", bear_c[0][14]);
        // dim 2: marginal effect advances the set (+1), centered scheme -> 1.0.
        assert!((bear_c[0][15] - 1.0).abs() < 1e-6, "got {}", bear_c[0][15]);
    }

    #[test]
    fn bear_c_marginal_regression_is_encoded_below_neutral() {
        // Board holds a size-1 and a size-2 group (progress 2). Placing a bear
        // that merges them into a size-3 drops both held sizes: progress 2 -> 1
        // (regression), which must read below the 0.5 neutral midpoint.
        let tokens = vec![
            placed(0, 0, 0, 0, Some(0)),
            placed(1, 0, 2, 0, Some(0)),
            placed(2, 0, 3, 0, Some(0)),
        ];
        // (1,0) bridges the lone bear (0,0) and the pair (2,0)/(3,0) -> size 4.
        let actions = vec![place_wildlife_action(0, 1, 0)];
        let record = root(0, tokens, actions);
        let bear_c = semantic_action_features(&record, cbddb_with('b', ScoringVariant::C)).unwrap();
        assert!((bear_c[0][14] - 2.0 / 3.0).abs() < 1e-6);
        // progress 2 -> 0 (a single size-4 group), delta -2 clamps to 0.0.
        assert!(bear_c[0][15] < 0.5, "got {}", bear_c[0][15]);
    }

    #[test]
    fn hawk_dims_are_los_typed_under_hawk_d() {
        // Existing hawk at (3,0), elk at (1,0); the action places a hawk at
        // (0,0), forming a line-of-sight pair with one species between.
        let tokens = vec![placed(0, 0, 3, 0, Some(3)), placed(1, 0, 1, 0, Some(1))];
        let actions = vec![place_wildlife_action(3, 0, 0)];
        let record = root(0, tokens, actions);

        let aaaaa = semantic_action_features(&record, ScoringCards::AAAAA).unwrap();
        // Card A: isolated (no adjacent hawk) -> 1.0, no adjacency penalty.
        assert_eq!(aaaaa[0][21], 1.0);
        assert_eq!(aaaaa[0][23], 0.0);

        let hawk_d = semantic_action_features(&record, cbddb_with('h', ScoringVariant::D)).unwrap();
        // Best reachable Hawk-D pair weight: one species between -> 4, /9.
        assert!((hawk_d[0][21] - 4.0 / 9.0).abs() < 1e-6, "got {}", hawk_d[0][21]);
        // One productive (>=1 species between) partner, /6.
        assert!((hawk_d[0][23] - 1.0 / 6.0).abs() < 1e-6, "got {}", hawk_d[0][23]);
        // The plain LOS partner count (general dim) is unchanged across cards.
        assert_eq!(aaaaa[0][22], hawk_d[0][22]);
    }

    #[test]
    fn fox_dim_counts_pair_types_under_fox_b() {
        // Fox placed at (0,0) with two bears and two elk adjacent: two neighbor
        // species appear as a PAIR (>=2), the Fox-B "2" tier.
        let tokens = vec![
            placed(0, 0, 1, 0, Some(0)),
            placed(1, 0, 1, -1, Some(0)),
            placed(2, 0, 0, -1, Some(1)),
            placed(3, 0, -1, 1, Some(1)),
        ];
        let actions = vec![place_wildlife_action(4, 0, 0)];
        let record = root(0, tokens, actions);

        let aaaaa = semantic_action_features(&record, ScoringCards::AAAAA).unwrap();
        // Card A fox_unique = distinct neighbor species (bear, elk) = 2, /5.
        assert!((aaaaa[0][24] - 2.0 / 5.0).abs() < 1e-6, "got {}", aaaaa[0][24]);

        let fox_b = semantic_action_features(&record, cbddb_with('f', ScoringVariant::B)).unwrap();
        // Fox B: two species with >=2 adjacent -> 2, /3.
        assert!((fox_b[0][24] - 2.0 / 3.0).abs() < 1e-6, "got {}", fox_b[0][24]);
    }

    #[test]
    fn elk_dim_reflects_shape_compactness_under_elk_b() {
        // Existing elk at (1,0) and (0,1); the action places an elk at (0,0)
        // forming a triangle (all three mutually adjacent) -> shape level 2.
        let tokens = vec![placed(0, 0, 1, 0, Some(1)), placed(1, 0, 0, 1, Some(1))];
        let actions = vec![place_wildlife_action(1, 0, 0)];
        let record = root(0, tokens, actions);

        let aaaaa = semantic_action_features(&record, ScoringCards::AAAAA).unwrap();
        // Card A straight-line length through (0,0): the (1,0)/(0,0) axis has
        // length 2 -> /4.
        assert!((aaaaa[0][16] - 2.0 / 4.0).abs() < 1e-6, "got {}", aaaaa[0][16]);

        let elk_b = semantic_action_features(&record, cbddb_with('e', ScoringVariant::B)).unwrap();
        // Elk B compactness: a triangle is level 2 -> /3.
        assert!((elk_b[0][16] - 2.0 / 3.0).abs() < 1e-6, "got {}", elk_b[0][16]);
    }

    #[test]
    fn coord_features_match_python_contract() {
        let canonical = json!({"kind": "canonical", "q": 3, "r": -1, "s": -2, "cell_index": 17});
        let features = coord_features(Some(&canonical));
        assert_eq!(features[0], normalizer(3.0, 6.0));
        assert_eq!(features[3], 1.0);
        assert_eq!(features[4], 0.0);
        assert_eq!(features[5], normalizer(17.0, 126.0));

        let overflow = json!({"kind": "overflow", "q": 7, "r": 0, "s": -7});
        let features = coord_features(Some(&overflow));
        assert_eq!(features[3], 0.0);
        assert_eq!(features[4], 1.0);
        assert_eq!(features[5], normalizer(-1.0, 126.0));
    }

    #[test]
    fn selected_action_index_finds_matching_action_id() {
        let record = json!({
            "selected_action": "b",
            "legal_actions": [{"action_id": "a"}, {"action_id": "b"}]
        });
        assert_eq!(selected_action_index(&record).unwrap(), 1);
    }
}

/// Action-row relation matrix for packed eval requests: `A x (T + A)` u8,
/// row-major, columns laid out token positions first then action positions
/// (unpadded). Row `r` holds the relation ids for edges whose SOURCE is
/// action `r` — exactly the rows the model's gated action bias consumes from
/// the dense matrix (`relation_ids[:, -action_count:, :]`).
pub fn action_relation_tail(
    root: &Value,
    token_count: usize,
    action_count: usize,
    cards: ScoringCards,
) -> Result<Vec<u8>> {
    const ACTION_USES_TILE_SLOT: u8 = 5;
    const ACTION_USES_WILDLIFE_SLOT: u8 = 6;
    const ACTION_TARGETS_TILE_FRONTIER: u8 = 7;
    const ACTION_TARGETS_WILDLIFE_CELL: u8 = 8;

    // Fast path equivalent to filtering `combined_relation_edges` down to
    // action-source rows: only the action loop there emits edges whose
    // source is an action position, so the token-token edge kinds
    // (same-owner board, adjacency, market pairing) never reach the tail
    // and their quadratic edge-map construction is skipped entirely.
    let seq_len = token_count + action_count;
    let tokens = field(root, "public_tokens")
        .and_then(|public_tokens| field(public_tokens, "tokens"))
        .and_then(Value::as_array)
        .context("root is missing public tokens")?;
    field(root, "public_tokens")
        .and_then(|public_tokens| field(public_tokens, "relations"))
        .and_then(Value::as_array)
        .context("root is missing public token relations")?;
    let active_seat = safe_i64(field(root, "active_seat"), 0);
    let mut market_tile: HashMap<i64, i32> = HashMap::new();
    let mut market_wildlife: HashMap<i64, i32> = HashMap::new();
    let mut active_frontier: HashMap<String, i32> = HashMap::new();
    let mut active_tile: HashMap<String, i32> = HashMap::new();

    for token in tokens {
        let index = safe_i64(field(token, "token_index"), -1) as i32;
        match field(token, "token_kind").and_then(Value::as_str) {
            Some("market_tile") => {
                market_tile.insert(safe_i64(field(token, "market_slot"), -1), index);
            }
            Some("market_wildlife") => {
                market_wildlife.insert(safe_i64(field(token, "market_slot"), -1), index);
            }
            Some("frontier") if safe_i64(field(token, "owner_seat"), -1) == active_seat => {
                if let Some(key) = relation_coord_key(field(token, "coord_ref")) {
                    active_frontier.insert(key, index);
                }
            }
            Some("placed_tile") if safe_i64(field(token, "owner_seat"), -1) == active_seat => {
                if let Some(key) = relation_coord_key(field(token, "coord_ref")) {
                    active_tile.insert(key, index);
                }
            }
            _ => {}
        }
    }

    let actions = field(root, "legal_actions")
        .and_then(Value::as_array)
        .context("root is missing legal_actions")?;
    let mut tail = vec![0u8; action_count * seq_len];
    for (action_index, action) in actions.iter().enumerate() {
        if action_index >= action_count {
            break;
        }
        let action_pos = (token_count + action_index) as i32;
        let tile_slot = safe_i64(
            field(action, "tile_slot").or_else(|| field(action, "draft_slot")),
            -1,
        );
        let wildlife_slot = safe_i64(
            field(action, "wildlife_slot").or_else(|| field(action, "draft_slot")),
            -1,
        );
        let target_frontier = relation_coord_key(field(action, "target_coord_ref"))
            .and_then(|key| active_frontier.get(&key).copied());
        let wildlife_target =
            relation_coord_key(field(action, "wildlife_coord_ref")).and_then(|key| {
                active_tile
                    .get(&key)
                    .or_else(|| active_frontier.get(&key))
                    .copied()
            });
        for (target, relation_id) in [
            (market_tile.get(&tile_slot).copied(), ACTION_USES_TILE_SLOT),
            (
                market_wildlife.get(&wildlife_slot).copied(),
                ACTION_USES_WILDLIFE_SLOT,
            ),
            (target_frontier, ACTION_TARGETS_TILE_FRONTIER),
            (wildlife_target, ACTION_TARGETS_WILDLIFE_CELL),
        ] {
            let Some(target) = target else {
                continue;
            };
            // Same guards as `set_relation` (later writes win, matching its
            // overwrite semantics through the write order here).
            if target < 0 || target >= seq_len as i32 || target == action_pos {
                continue;
            }
            tail[action_index * seq_len + target as usize] = relation_id;
            // The combined edge list also stores the mirrored
            // `(target, action_pos)` edge; it reaches the tail only when the
            // target itself sits in the action range (never true for
            // well-formed token indexes, mirrored here for exactness).
            if target as usize >= token_count {
                tail[(target as usize - token_count) * seq_len + action_pos as usize] = relation_id;
            }
        }
    }

    // Action-source hawk line-of-sight edges (Hawk C/D only). Same shared
    // geometry as `combined_relation_edges`, written AFTER the ids-5..8 loop so a
    // LOS id overwrites any 5..8 id already at the same (action_row, partner)
    // cell — reproducing that path's last-write-wins overwrite for byte parity.
    for (action_index, partner_node, relation_id) in
        hawk_los_action_edges(root, active_seat, cards)
    {
        if action_index >= action_count {
            continue;
        }
        let action_pos = (token_count + action_index) as i32;
        let target = partner_node;
        // Same guards as `set_relation` (out-of-range and self edges dropped).
        if target < 0 || target >= seq_len as i32 || target == action_pos {
            continue;
        }
        let relation_id = relation_id as u8;
        tail[action_index * seq_len + target as usize] = relation_id;
        // Mirror `(partner, action_pos)` reaches the tail only if the partner is
        // itself in the action range (never true for a real hawk token index,
        // mirrored here for exactness with `combined_relation_edges`).
        if target as usize >= token_count {
            tail[(target as usize - token_count) * seq_len + action_pos as usize] = relation_id;
        }
    }
    Ok(tail)
}

/// Reference implementation of `action_relation_tail` via the full combined
/// edge list; kept for the equivalence test.
#[cfg(test)]
pub fn action_relation_tail_reference(
    root: &Value,
    token_count: usize,
    action_count: usize,
    cards: ScoringCards,
) -> Result<Vec<u8>> {
    let seq_len = token_count + action_count;
    // The action tail carries only action-source edges. Under AAAAA the sole
    // action-source edges are ids 5..8; under Hawk C/D the shared
    // `hawk_los_action_edges` also lands action-source LOS ids (13..16), while
    // token->token hawk LOS edges (ids 9..12) are sliced away by the source
    // range guard below. Reference is parameterized by `cards` so both rulesets
    // are checked against the fast path.
    let edges = combined_relation_edges(root, token_count, action_count, cards)?;
    let mut tail = vec![0u8; action_count * seq_len];
    for [source, target, relation_id] in edges {
        let source = source as usize;
        let target = target as usize;
        if source < token_count || source >= seq_len || target >= seq_len {
            continue;
        }
        let row = source - token_count;
        tail[row * seq_len + target] = relation_id.clamp(0, 255) as u8;
    }
    Ok(tail)
}
