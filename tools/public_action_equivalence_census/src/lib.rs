use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    error::Error,
    fs::{self, File},
    io::{BufReader, Read, Write},
    path::{Path, PathBuf},
};

use blake3::Hasher;
use cascadia_data::{
    DatasetSplit, ExactSemanticSupply, GradedOracleDatasetManifest, GradedOracleGroup,
    PositionRecord, read_graded_oracle_shard, validate_graded_oracle_dataset,
};
use cascadia_game::{
    DraftChoice, GameConfig, GameSeed, GameState, MarketPrelude, MarketSlot, TilePlacement,
    TurnAction, WildlifeWipe,
};
use r3_action_edit_census::{ActionEdit, PublicStateTrunk, SupplySnapshot};
use serde::{Deserialize, Serialize};

pub const SCHEMA_VERSION: u16 = 1;
pub const EXPERIMENT_ID: &str = "s7-public-action-equivalence-foundation-v2";
pub const PROTOCOL_ID: &str = "s7-exact-semantic-transition-v2";
pub const CLASSIFICATION_INVALID: &str = "public_action_equivalence_invalid";
pub const CLASSIFICATION_FUTILE: &str = "public_action_equivalence_proof_only_futile";
pub const CLASSIFICATION_PROMISING: &str = "public_action_equivalence_promising";

const ACTION_HASH_DOMAIN: &[u8] = b"cascadia-v2-full-legal-action-v1";
const SEMANTIC_KEY_DOMAIN: &[u8] = b"cascadia-v2-s7-semantic-state-supply-v1";
const SAFE_KEY_DOMAIN: &[u8] = b"cascadia-v2-s7-serving-safe-equivalence-v1";
const TRACE_DOMAIN: &[u8] = b"cascadia-v2-s7-hidden-effect-trace-v1";
const REPORT_DOMAIN: &[u8] = b"cascadia-v2-s7-report-v1";
const SOURCE_DIGEST_BYTES: usize = 64;
const PROMOTION_MEDIAN_PPM: u64 = 20_000;
const PROMOTION_TAIL_PPM: u64 = 50_000;
const PROMOTION_TAIL_ABSOLUTE: u64 = 128;

#[derive(Debug, Clone)]
pub struct CensusConfig {
    pub dataset_roots: Vec<PathBuf>,
    pub shard_index: u8,
    pub shard_count: u8,
    pub source_bundle_blake3: String,
    pub maximum_selected_groups: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DatasetIdentity {
    pub split: String,
    pub root: String,
    pub dataset_id: String,
    pub manifest_blake3: String,
    pub groups: usize,
    pub candidates: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CensusChecks {
    pub selected_groups: usize,
    pub selected_candidates: usize,
    pub position_record_checks: usize,
    pub public_state_hash_checks: usize,
    pub public_supply_checks: usize,
    pub canonical_action_hash_checks: usize,
    pub grouped_r3_action_matches: usize,
    pub r3_apply_checks: usize,
    pub authoritative_collision_checks: usize,
    pub authoritative_r3_record_checks: usize,
    pub authoritative_supply_checks: usize,
    pub semantic_successor_parity_classes: usize,
    pub strict_successor_parity_classes: usize,
    pub invariant_failures: usize,
}

impl CensusChecks {
    fn add_assign(&mut self, other: &Self) {
        self.selected_groups += other.selected_groups;
        self.selected_candidates += other.selected_candidates;
        self.position_record_checks += other.position_record_checks;
        self.public_state_hash_checks += other.public_state_hash_checks;
        self.public_supply_checks += other.public_supply_checks;
        self.canonical_action_hash_checks += other.canonical_action_hash_checks;
        self.grouped_r3_action_matches += other.grouped_r3_action_matches;
        self.r3_apply_checks += other.r3_apply_checks;
        self.authoritative_collision_checks += other.authoritative_collision_checks;
        self.authoritative_r3_record_checks += other.authoritative_r3_record_checks;
        self.authoritative_supply_checks += other.authoritative_supply_checks;
        self.semantic_successor_parity_classes += other.semantic_successor_parity_classes;
        self.strict_successor_parity_classes += other.strict_successor_parity_classes;
        self.invariant_failures += other.invariant_failures;
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClassStats {
    pub candidates: usize,
    pub unique_classes: usize,
    pub collapsed_candidates: usize,
    pub duplicate_classes: usize,
    pub candidates_in_duplicate_classes: usize,
    pub pair_collisions: u64,
    pub maximum_class_size: usize,
    pub reduction_ppm: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct LabelSpread {
    pub compared_classes: usize,
    pub maximum_r600_mean_spread: f64,
    pub maximum_r1200_mean_spread: f64,
    pub maximum_r4800_mean_spread: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GroupRecord {
    pub split: String,
    pub row: usize,
    pub group_id: u64,
    pub raw_seed: u64,
    pub completed_turns: u16,
    pub candidates: usize,
    pub semantic_state_supply: ClassStats,
    pub serving_safe: ClassStats,
    pub exact_public_within_safe: ClassStats,
    pub exact_hidden_successor_within_safe: ClassStats,
    pub trace_rejected_collapses: usize,
    pub semantic_identity_collapses_beyond_exact_public: usize,
    pub selected_safe_class_size: usize,
    pub champion_safe_class_size: usize,
    pub label_spread: LabelSpread,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CensusScientific {
    pub schema_version: u16,
    pub experiment_id: String,
    pub protocol_id: String,
    pub source_bundle_blake3: String,
    pub executable_blake3: String,
    pub shard_index: u8,
    pub shard_count: u8,
    pub complete_open_shard: bool,
    pub partition_rule: String,
    pub datasets: Vec<DatasetIdentity>,
    pub checks: CensusChecks,
    pub records: Vec<GroupRecord>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CensusReport {
    pub report_id: String,
    pub scientific: CensusScientific,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdversarialScientific {
    pub schema_version: u16,
    pub experiment_id: String,
    pub protocol_id: String,
    pub synthetic_states: usize,
    pub exhaustive_legal_actions: usize,
    pub positive_duplicate_witnesses: usize,
    pub ordered_trace_rejections: usize,
    pub semantic_class_checks: usize,
    pub strict_class_checks: usize,
    pub failures: Vec<String>,
    pub passed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdversarialReport {
    pub report_id: String,
    pub scientific: AdversarialScientific,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DuplicateSmokeScientific {
    pub schema_version: u16,
    pub experiment_id: String,
    pub protocol_id: String,
    pub dataset_id: String,
    pub manifest_blake3: String,
    pub split: String,
    pub group_id: u64,
    pub original_candidates: usize,
    pub duplicated_candidate_index: usize,
    pub checks: CensusChecks,
    pub record: GroupRecord,
    pub passed: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DuplicateSmokeReport {
    pub report_id: String,
    pub scientific: DuplicateSmokeScientific,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReductionDistribution {
    pub groups: usize,
    pub candidates: usize,
    pub unique_classes: usize,
    pub collapsed_candidates: usize,
    pub weighted_reduction_ppm: u64,
    pub median_reduction_ppm: u64,
    pub p90_reduction_ppm: u64,
    pub p99_reduction_ppm: u64,
    pub maximum_reduction_ppm: u64,
    pub median_collapsed_candidates: u64,
    pub p90_collapsed_candidates: u64,
    pub p99_collapsed_candidates: u64,
    pub maximum_collapsed_candidates: u64,
    pub groups_with_any_collapse: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromotionGates {
    pub adversarial_pass: bool,
    pub complete_open_corpus: bool,
    pub zero_invariant_failures: bool,
    pub semantic_successor_parity: bool,
    pub strict_successor_parity: bool,
    pub validation_median_reduction_pass: bool,
    pub validation_tail_reduction_pass: bool,
    pub promotion_pass: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AggregateScientific {
    pub schema_version: u16,
    pub experiment_id: String,
    pub protocol_id: String,
    pub source_bundle_blake3: String,
    pub executable_blake3: String,
    pub valid: bool,
    pub classification: String,
    pub datasets: Vec<DatasetIdentity>,
    pub checks: CensusChecks,
    pub train_semantic_state_supply: ReductionDistribution,
    pub train_serving_safe: ReductionDistribution,
    pub train_exact_public_within_safe: ReductionDistribution,
    pub validation_semantic_state_supply: ReductionDistribution,
    pub validation_serving_safe: ReductionDistribution,
    pub validation_exact_public_within_safe: ReductionDistribution,
    pub validation_trace_rejected_collapses: usize,
    pub validation_semantic_identity_collapses_beyond_exact_public: usize,
    pub gates: PromotionGates,
    pub records: Vec<GroupRecord>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AggregateReport {
    pub report_id: String,
    pub scientific: AggregateScientific,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OrderProofScientific {
    pub schema_version: u16,
    pub experiment_id: String,
    pub protocol_id: String,
    pub forward_report_id: String,
    pub reverse_report_id: String,
    pub forward_scientific_blake3: String,
    pub reverse_scientific_blake3: String,
    pub byte_identical: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OrderProof {
    pub report_id: String,
    pub scientific: OrderProofScientific,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct DraftBatchKey {
    replace_three_of_a_kind: bool,
    wipe_masks: Vec<u8>,
    draft_kind: u8,
    tile_slot: u8,
    wildlife_slot: u8,
}

#[derive(Debug, Clone)]
struct CandidateEvidence {
    action: TurnAction,
    action_hash: [u8; 32],
    semantic_material: Vec<u8>,
    safe_material: Vec<u8>,
    record_bytes: Vec<u8>,
    supply_bytes: Vec<u8>,
    trace_bytes: Vec<u8>,
    r600: Option<f64>,
    r1200: Option<f64>,
    r4800: Option<f64>,
}

#[derive(Debug)]
struct CollisionEvidence {
    exact_public_material: Vec<u8>,
    exact_successor_material: Vec<u8>,
    semantic_successor_material: Vec<u8>,
}

pub fn run_census(config: &CensusConfig) -> Result<CensusReport, Box<dyn Error>> {
    validate_census_config(config)?;
    let executable_blake3 = checksum_file(&std::env::current_exe()?)?;
    let mut datasets = Vec::new();
    let mut records = Vec::new();
    let mut checks = CensusChecks::default();
    let mut remaining = config.maximum_selected_groups.unwrap_or(usize::MAX);

    for root in &config.dataset_roots {
        let manifest_path = root.join("dataset.json");
        let manifest_bytes = fs::read(&manifest_path)?;
        let manifest: GradedOracleDatasetManifest = serde_json::from_slice(&manifest_bytes)?;
        validate_graded_oracle_dataset(root, &manifest)?;
        if !matches!(
            manifest.split,
            DatasetSplit::Train | DatasetSplit::Validation
        ) {
            return Err(format!(
                "S7 refuses sealed {} data at {}",
                manifest.split.id(),
                root.display()
            )
            .into());
        }
        if manifest.game != GameConfig::research_aaaaa(4)? {
            return Err("S7 dataset does not use the frozen four-player AAAAA ruleset".into());
        }
        datasets.push(DatasetIdentity {
            split: manifest.split.id().to_owned(),
            root: root
                .file_name()
                .and_then(|name| name.to_str())
                .ok_or("S7 dataset root has no UTF-8 terminal component")?
                .to_owned(),
            dataset_id: manifest.dataset_id.clone(),
            manifest_blake3: blake3::hash(&manifest_bytes).to_hex().to_string(),
            groups: manifest.total_groups,
            candidates: manifest.total_records,
        });
        census_dataset(
            root,
            &manifest,
            config.shard_index,
            config.shard_count,
            &mut remaining,
            &mut records,
            &mut checks,
        )?;
    }
    datasets.sort_by(|left, right| left.split.cmp(&right.split));
    records.sort_by(|left, right| {
        (split_order(&left.split), left.row).cmp(&(split_order(&right.split), right.row))
    });
    let expected_selected = datasets
        .iter()
        .map(|dataset| {
            selected_row_count(
                dataset.groups,
                usize::from(config.shard_index),
                usize::from(config.shard_count),
            )
        })
        .sum::<usize>();
    let complete_open_shard =
        config.maximum_selected_groups.is_none() && records.len() == expected_selected;
    if config.maximum_selected_groups.is_none() && !complete_open_shard {
        return Err(format!(
            "S7 shard covered {} groups instead of {expected_selected}",
            records.len()
        )
        .into());
    }
    if records.is_empty() {
        return Err("S7 shard selected no groups".into());
    }
    if checks.selected_groups != records.len()
        || checks.selected_candidates
            != records
                .iter()
                .map(|record| record.candidates)
                .sum::<usize>()
        || checks.invariant_failures != 0
    {
        return Err("S7 census check totals are inconsistent".into());
    }

    let scientific = CensusScientific {
        schema_version: SCHEMA_VERSION,
        experiment_id: EXPERIMENT_ID.to_owned(),
        protocol_id: PROTOCOL_ID.to_owned(),
        source_bundle_blake3: config.source_bundle_blake3.clone(),
        executable_blake3,
        shard_index: config.shard_index,
        shard_count: config.shard_count,
        complete_open_shard,
        partition_rule: "(split_row % shard_count) == shard_index".to_owned(),
        datasets,
        checks,
        records,
    };
    Ok(CensusReport {
        report_id: report_id(&scientific)?,
        scientific,
    })
}

fn validate_census_config(config: &CensusConfig) -> Result<(), Box<dyn Error>> {
    if config.dataset_roots.is_empty() {
        return Err("at least one --dataset-root is required".into());
    }
    if config.shard_count == 0 || config.shard_index >= config.shard_count {
        return Err("S7 shard index/count is invalid".into());
    }
    validate_lower_hex_digest(
        "source bundle",
        &config.source_bundle_blake3,
        SOURCE_DIGEST_BYTES,
    )?;
    let roots = config
        .dataset_roots
        .iter()
        .map(|path| path.display().to_string())
        .collect::<BTreeSet<_>>();
    if roots.len() != config.dataset_roots.len() {
        return Err("S7 dataset roots are duplicated".into());
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn census_dataset(
    root: &Path,
    manifest: &GradedOracleDatasetManifest,
    shard_index: u8,
    shard_count: u8,
    remaining: &mut usize,
    records: &mut Vec<GroupRecord>,
    checks: &mut CensusChecks,
) -> Result<(), Box<dyn Error>> {
    let split = manifest.split;
    let mut row = 0usize;
    for shard in &manifest.shards {
        let groups = read_graded_oracle_shard(root, split, shard)?;
        let mut game = GameState::new(
            GameConfig::research_aaaaa(4)?,
            GameSeed::from_u64(shard.first_game_index),
        )?;
        for group in groups {
            verify_group_parent(&game, &group, checks)?;
            let selected =
                row % usize::from(shard_count) == usize::from(shard_index) && *remaining > 0;
            if selected {
                let record = analyze_group(split.id(), row, &game, &group, checks)?;
                records.push(record);
                checks.selected_groups += 1;
                checks.selected_candidates += group.candidates.len();
                *remaining = remaining.saturating_sub(1);
            }
            let champion = group.candidates[usize::from(group.champion_index)]
                .action
                .to_game_action(&game)?;
            game.apply(&champion)?;
            row += 1;
        }
        if !game.is_game_over() {
            return Err(format!(
                "S7 source game {} did not contain 80 complete decisions",
                shard.first_game_index
            )
            .into());
        }
    }
    if row != manifest.total_groups {
        return Err(format!(
            "S7 replayed {row} {} groups instead of {}",
            split.id(),
            manifest.total_groups
        )
        .into());
    }
    Ok(())
}

fn verify_group_parent(
    game: &GameState,
    group: &GradedOracleGroup,
    checks: &mut CensusChecks,
) -> Result<(), Box<dyn Error>> {
    if game.completed_turns() != group.completed_turns
        || game.current_player() != usize::from(group.current_player)
        || PositionRecord::observe(game, group.raw_seed).to_bytes() != group.position.to_bytes()
    {
        return Err(format!("S7 graded replay drifted at group {}", group.group_id).into());
    }
    checks.position_record_checks += 1;
    if *game.public_state().canonical_hash().as_bytes() != group.public_state_hash {
        return Err(format!("S7 public-state hash drifted at group {}", group.group_id).into());
    }
    checks.public_state_hash_checks += 1;
    if game.public_supply() != group.public_supply {
        return Err(format!("S7 public supply drifted at group {}", group.group_id).into());
    }
    checks.public_supply_checks += 1;
    Ok(())
}

fn analyze_group(
    split: &str,
    row: usize,
    game: &GameState,
    group: &GradedOracleGroup,
    checks: &mut CensusChecks,
) -> Result<GroupRecord, Box<dyn Error>> {
    let candidate_count = group.candidates.len();
    let trunk = PublicStateTrunk::observe(game, group.raw_seed)?;
    let prepared = trunk.prepare_action_edits()?;
    let mut actions = Vec::with_capacity(candidate_count);
    let mut batches = BTreeMap::<DraftBatchKey, Vec<(usize, TurnAction)>>::new();
    for (index, candidate) in group.candidates.iter().enumerate() {
        let action = candidate.action.to_game_action(game)?;
        if canonical_action_hash(&action)? != candidate.action_hash {
            return Err(format!(
                "S7 action hash drifted at group {} candidate {index}",
                group.group_id
            )
            .into());
        }
        checks.canonical_action_hash_checks += 1;
        batches
            .entry(draft_batch_key(&action))
            .or_default()
            .push((index, action.clone()));
        actions.push(action);
    }

    let mut edits = vec![None; candidate_count];
    for targets in batches.into_values() {
        let prelude = targets[0].1.prelude();
        let draft = targets[0].1.draft;
        let mut target_positions = HashMap::<TurnAction, Vec<usize>>::new();
        for (position, action) in targets {
            target_positions.entry(action).or_default().push(position);
        }
        for (action, edit) in prepared.observe_draft_actions(game, &prelude, draft)? {
            if let Some(positions) = target_positions.get(&action) {
                for position in positions {
                    if edits[*position].replace(edit.clone()).is_some() {
                        return Err(format!(
                            "S7 R3 enumeration duplicated group {} candidate {position}",
                            group.group_id
                        )
                        .into());
                    }
                    checks.grouped_r3_action_matches += 1;
                }
            }
        }
    }
    if edits.iter().any(Option::is_none) {
        return Err(format!("S7 R3 enumeration missed group {}", group.group_id).into());
    }

    let mut evidence = Vec::with_capacity(candidate_count);
    let mut semantic_map = BTreeMap::<[u8; 32], Vec<usize>>::new();
    let mut safe_map = BTreeMap::<[u8; 32], Vec<usize>>::new();
    for (index, action) in actions.into_iter().enumerate() {
        let applied = prepared.apply(edits[index].as_ref().expect("checked complete"))?;
        checks.r3_apply_checks += 1;
        let mut record = applied.record;
        record.game_index = 0;
        record.targets.fill(0);
        let record_bytes = record.to_bytes().to_vec();
        let supply_bytes = applied.supply.canonical_bytes()?;
        let trace_bytes = hidden_effect_trace(&action);
        let semantic_material =
            framed_material(SEMANTIC_KEY_DOMAIN, [&record_bytes[..], &supply_bytes[..]]);
        let safe_material = framed_material(
            SAFE_KEY_DOMAIN,
            [&record_bytes[..], &supply_bytes[..], &trace_bytes[..]],
        );
        semantic_map
            .entry(*blake3::hash(&semantic_material).as_bytes())
            .or_default()
            .push(index);
        safe_map
            .entry(*blake3::hash(&safe_material).as_bytes())
            .or_default()
            .push(index);
        let candidate = &group.candidates[index];
        evidence.push(CandidateEvidence {
            action,
            action_hash: candidate.action_hash,
            semantic_material,
            safe_material,
            record_bytes,
            supply_bytes,
            trace_bytes,
            r600: estimate_mean(candidate.r600.mean, candidate.r600.samples),
            r1200: estimate_mean(candidate.r1200.mean, candidate.r1200.samples),
            r4800: estimate_mean(candidate.r4800.mean, candidate.r4800.samples),
        });
    }

    validate_hashed_material_classes(&semantic_map, &evidence, |candidate| {
        &candidate.semantic_material
    })?;
    validate_hashed_material_classes(&safe_map, &evidence, |candidate| &candidate.safe_material)?;
    let semantic_state_supply = class_stats_from_map(candidate_count, &semantic_map);
    let serving_safe = class_stats_from_map(candidate_count, &safe_map);
    let trace_rejected_collapses = semantic_state_supply
        .collapsed_candidates
        .checked_sub(serving_safe.collapsed_candidates)
        .ok_or("S7 safe key collapsed more actions than its semantic parent")?;

    let mut exact_public_sizes = Vec::with_capacity(candidate_count);
    let mut exact_successor_sizes = Vec::with_capacity(candidate_count);
    let mut label_spread = LabelSpread::default();
    let mut semantic_identity_collapses_beyond_exact_public = 0usize;
    let mut selected_safe_class_size = 1usize;
    let mut champion_safe_class_size = 1usize;

    for members in safe_map.values() {
        if members.contains(&usize::from(group.selected_index)) {
            selected_safe_class_size = members.len();
        }
        if members.contains(&usize::from(group.champion_index)) {
            champion_safe_class_size = members.len();
        }
        if members.len() == 1 {
            exact_public_sizes.push(1);
            exact_successor_sizes.push(1);
            continue;
        }
        validate_safe_material(members, &evidence)?;
        update_label_spread(&mut label_spread, members, &evidence);
        let mut collisions = Vec::with_capacity(members.len());
        for member in members {
            collisions.push(authoritative_collision_evidence(
                game,
                &evidence[*member],
                checks,
            )?);
        }
        let exact_public_map = group_materials(
            members,
            collisions
                .iter()
                .map(|collision| &collision.exact_public_material),
        )?;
        let exact_successor_map = group_materials(
            members,
            collisions
                .iter()
                .map(|collision| &collision.exact_successor_material),
        )?;
        for class in exact_public_map.values() {
            exact_public_sizes.push(class.len());
            if class.len() > 1 {
                let expected = &collisions[class[0]].exact_successor_material;
                if class
                    .iter()
                    .any(|relative| collisions[*relative].exact_successor_material != *expected)
                {
                    return Err(format!(
                        "S7 exact public class changed hidden successor at group {}",
                        group.group_id
                    )
                    .into());
                }
                checks.strict_successor_parity_classes += 1;
            }
        }
        for class in exact_successor_map.values() {
            exact_successor_sizes.push(class.len());
        }
        let expected_semantic_successor = &collisions[0].semantic_successor_material;
        if collisions
            .iter()
            .any(|collision| collision.semantic_successor_material != *expected_semantic_successor)
        {
            return Err(format!(
                "S7 serving-safe class changed semantic successor at group {}",
                group.group_id
            )
            .into());
        }
        checks.semantic_successor_parity_classes += 1;
        semantic_identity_collapses_beyond_exact_public +=
            collapses_beyond_exact_public(members.len(), exact_public_map.len())?;
    }

    let exact_public_within_safe = class_stats_from_sizes(&exact_public_sizes);
    let exact_hidden_successor_within_safe = class_stats_from_sizes(&exact_successor_sizes);
    if exact_public_within_safe.candidates != candidate_count
        || exact_hidden_successor_within_safe.candidates != candidate_count
        || semantic_identity_collapses_beyond_exact_public
            != serving_safe
                .collapsed_candidates
                .checked_sub(exact_public_within_safe.collapsed_candidates)
                .ok_or("S7 exact-public reduction exceeds serving-safe reduction")?
    {
        return Err(format!("S7 class accounting drifted at group {}", group.group_id).into());
    }

    Ok(GroupRecord {
        split: split.to_owned(),
        row,
        group_id: group.group_id,
        raw_seed: group.raw_seed,
        completed_turns: group.completed_turns,
        candidates: candidate_count,
        semantic_state_supply,
        serving_safe,
        exact_public_within_safe,
        exact_hidden_successor_within_safe,
        trace_rejected_collapses,
        semantic_identity_collapses_beyond_exact_public,
        selected_safe_class_size,
        champion_safe_class_size,
        label_spread,
    })
}

fn validate_safe_material(
    members: &[usize],
    evidence: &[CandidateEvidence],
) -> Result<(), Box<dyn Error>> {
    let first = &evidence[members[0]];
    for member in members.iter().copied().skip(1) {
        let candidate = &evidence[member];
        if candidate.safe_material != first.safe_material
            || candidate.record_bytes != first.record_bytes
            || candidate.supply_bytes != first.supply_bytes
            || candidate.trace_bytes != first.trace_bytes
        {
            return Err("S7 safe-key hash collision or material mismatch".into());
        }
    }
    Ok(())
}

fn validate_hashed_material_classes(
    classes: &BTreeMap<[u8; 32], Vec<usize>>,
    evidence: &[CandidateEvidence],
    material: impl Fn(&CandidateEvidence) -> &Vec<u8>,
) -> Result<(), Box<dyn Error>> {
    for members in classes.values().filter(|members| members.len() > 1) {
        let expected = material(&evidence[members[0]]);
        if members
            .iter()
            .copied()
            .skip(1)
            .any(|member| material(&evidence[member]) != expected)
        {
            return Err("S7 BLAKE3 class contains unequal canonical material".into());
        }
    }
    Ok(())
}

fn authoritative_collision_evidence(
    game: &GameState,
    evidence: &CandidateEvidence,
    checks: &mut CensusChecks,
) -> Result<CollisionEvidence, Box<dyn Error>> {
    if canonical_action_hash(&evidence.action)? != evidence.action_hash {
        return Err("S7 collision action hash drifted".into());
    }
    let public = game.preview_public_afterstate(&evidence.action)?;
    checks.authoritative_collision_checks += 1;
    let mut authoritative_record =
        PositionRecord::observe_public_for_seat(&public, 0, game.current_player());
    authoritative_record.targets.fill(0);
    if authoritative_record.to_bytes().as_slice() != evidence.record_bytes {
        return Err("S7 authoritative public record differs from exact R3 apply".into());
    }
    checks.authoritative_r3_record_checks += 1;
    let authoritative_supply =
        SupplySnapshot::from_exact(&ExactSemanticSupply::from_public_state(&public)?);
    if authoritative_supply.canonical_bytes()? != evidence.supply_bytes {
        return Err("S7 authoritative semantic supply differs from exact R3 apply".into());
    }
    checks.authoritative_supply_checks += 1;

    let exact_public_material = framed_material(
        b"cascadia-v2-s7-exact-public-v1",
        [
            public.canonical_bytes().as_slice(),
            evidence.supply_bytes.as_slice(),
            evidence.trace_bytes.as_slice(),
        ],
    );
    let successor = game.transition(&evidence.action)?;
    let exact_successor_material = successor.canonical_bytes();
    let mut successor_record =
        PositionRecord::observe_for_seat(&successor, 0, successor.current_player());
    successor_record.targets.fill(0);
    let successor_supply = ExactSemanticSupply::from_game(&successor)?;
    let semantic_successor_material = framed_material(
        b"cascadia-v2-s7-semantic-successor-v1",
        [
            successor_record.to_bytes().as_slice(),
            successor_supply.canonical_bytes().as_slice(),
        ],
    );
    Ok(CollisionEvidence {
        exact_public_material,
        exact_successor_material,
        semantic_successor_material,
    })
}

fn group_materials<'a>(
    members: &[usize],
    materials: impl Iterator<Item = &'a Vec<u8>>,
) -> Result<BTreeMap<Vec<u8>, Vec<usize>>, Box<dyn Error>> {
    let materials = materials.collect::<Vec<_>>();
    if materials.len() != members.len() {
        return Err("S7 collision material count drifted".into());
    }
    let mut grouped = BTreeMap::<Vec<u8>, Vec<usize>>::new();
    for (relative, material) in materials.into_iter().enumerate() {
        grouped.entry(material.clone()).or_default().push(relative);
    }
    Ok(grouped)
}

fn collapses_beyond_exact_public(
    serving_safe_class_size: usize,
    exact_public_class_count: usize,
) -> Result<usize, Box<dyn Error>> {
    if serving_safe_class_size < 2
        || exact_public_class_count == 0
        || exact_public_class_count > serving_safe_class_size
    {
        return Err("S7 semantic/exact-public class accounting is invalid".into());
    }
    Ok(exact_public_class_count - 1)
}

fn update_label_spread(
    summary: &mut LabelSpread,
    members: &[usize],
    evidence: &[CandidateEvidence],
) {
    summary.compared_classes += 1;
    summary.maximum_r600_mean_spread =
        summary
            .maximum_r600_mean_spread
            .max(optional_spread(members, evidence, |candidate| {
                candidate.r600
            }));
    summary.maximum_r1200_mean_spread =
        summary
            .maximum_r1200_mean_spread
            .max(optional_spread(members, evidence, |candidate| {
                candidate.r1200
            }));
    summary.maximum_r4800_mean_spread =
        summary
            .maximum_r4800_mean_spread
            .max(optional_spread(members, evidence, |candidate| {
                candidate.r4800
            }));
}

fn optional_spread(
    members: &[usize],
    evidence: &[CandidateEvidence],
    value: impl Fn(&CandidateEvidence) -> Option<f64>,
) -> f64 {
    let values = members
        .iter()
        .filter_map(|member| value(&evidence[*member]))
        .collect::<Vec<_>>();
    if values.len() < 2 {
        return 0.0;
    }
    let minimum = values.iter().copied().fold(f64::INFINITY, f64::min);
    let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    maximum - minimum
}

fn estimate_mean(mean: f32, samples: u16) -> Option<f64> {
    (samples > 0).then_some(f64::from(mean))
}

fn class_stats_from_map<K: Ord>(
    candidates: usize,
    classes: &BTreeMap<K, Vec<usize>>,
) -> ClassStats {
    class_stats_from_sizes(&classes.values().map(Vec::len).collect::<Vec<_>>())
        .with_expected_candidates(candidates)
}

fn class_stats_from_sizes(sizes: &[usize]) -> ClassStats {
    let candidates = sizes.iter().sum::<usize>();
    let unique_classes = sizes.len();
    let collapsed_candidates = candidates.saturating_sub(unique_classes);
    let duplicate_classes = sizes.iter().filter(|size| **size > 1).count();
    let candidates_in_duplicate_classes = sizes.iter().filter(|size| **size > 1).sum::<usize>();
    let pair_collisions = sizes
        .iter()
        .map(|size| {
            let size = *size as u64;
            size.saturating_mul(size.saturating_sub(1)) / 2
        })
        .sum();
    let maximum_class_size = sizes.iter().copied().max().unwrap_or(0);
    ClassStats {
        candidates,
        unique_classes,
        collapsed_candidates,
        duplicate_classes,
        candidates_in_duplicate_classes,
        pair_collisions,
        maximum_class_size,
        reduction_ppm: ratio_ppm(collapsed_candidates as u64, candidates as u64),
    }
}

trait ExpectedCandidates {
    fn with_expected_candidates(self, expected: usize) -> Self;
}

impl ExpectedCandidates for ClassStats {
    fn with_expected_candidates(self, expected: usize) -> Self {
        assert_eq!(self.candidates, expected);
        self
    }
}

fn hidden_effect_trace(action: &TurnAction) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(16 + action.wildlife_wipes.len());
    bytes.extend_from_slice(TRACE_DOMAIN);
    bytes.push(u8::from(action.replace_three_of_a_kind));
    bytes.push(action.wildlife_wipes.len() as u8);
    for wipe in &action.wildlife_wipes {
        bytes.push(wipe_mask(wipe));
    }
    match action.draft {
        DraftChoice::Paired { slot } => {
            bytes.extend_from_slice(&[0, slot.index() as u8, slot.index() as u8]);
        }
        DraftChoice::Independent {
            tile_slot,
            wildlife_slot,
        } => {
            bytes.extend_from_slice(&[1, tile_slot.index() as u8, wildlife_slot.index() as u8]);
        }
    }
    bytes.push(u8::from(action.wildlife.is_none()));
    bytes
}

fn wipe_mask(wipe: &WildlifeWipe) -> u8 {
    wipe.slots
        .iter()
        .fold(0u8, |mask, slot| mask | (1 << slot.index()))
}

fn draft_batch_key(action: &TurnAction) -> DraftBatchKey {
    let (draft_kind, tile_slot, wildlife_slot) = match action.draft {
        DraftChoice::Paired { slot } => (0, slot.index() as u8, slot.index() as u8),
        DraftChoice::Independent {
            tile_slot,
            wildlife_slot,
        } => (1, tile_slot.index() as u8, wildlife_slot.index() as u8),
    };
    DraftBatchKey {
        replace_three_of_a_kind: action.replace_three_of_a_kind,
        wipe_masks: action.wildlife_wipes.iter().map(wipe_mask).collect(),
        draft_kind,
        tile_slot,
        wildlife_slot,
    }
}

fn canonical_action_hash(action: &TurnAction) -> Result<[u8; 32], Box<dyn Error>> {
    let mut hasher = Hasher::new();
    hasher.update(ACTION_HASH_DOMAIN);
    hasher.update(&serde_json::to_vec(action)?);
    Ok(*hasher.finalize().as_bytes())
}

fn framed_material<'a>(domain: &[u8], parts: impl IntoIterator<Item = &'a [u8]>) -> Vec<u8> {
    let parts = parts.into_iter().collect::<Vec<_>>();
    let capacity = domain.len() + 8 + parts.iter().map(|part| 8 + part.len()).sum::<usize>();
    let mut bytes = Vec::with_capacity(capacity);
    bytes.extend_from_slice(domain);
    bytes.extend_from_slice(&(parts.len() as u64).to_le_bytes());
    for part in parts {
        bytes.extend_from_slice(&(part.len() as u64).to_le_bytes());
        bytes.extend_from_slice(part);
    }
    bytes
}

pub fn run_adversarial_suite() -> Result<AdversarialReport, Box<dyn Error>> {
    let mut failures = Vec::new();
    let mut exhaustive_legal_actions = 0usize;
    let mut positive_duplicate_witnesses = 0usize;
    let mut semantic_class_checks = 0usize;
    let mut strict_class_checks = 0usize;

    for seed in [17u64, 811, 65_537] {
        let game = GameState::new(GameConfig::research_aaaaa(4)?, GameSeed::from_u64(seed))?;
        let trunk = PublicStateTrunk::observe(&game, seed)?;
        let legal = ActionEdit::observe_legal_actions(&game, &trunk, &MarketPrelude::default())?;
        exhaustive_legal_actions += legal.len();
        let mut rows = legal;
        if let Some(first) = rows.first().cloned() {
            rows.push(first);
            positive_duplicate_witnesses += 1;
        }
        if let Err(error) = validate_adversarial_state(&game, &trunk, &rows) {
            failures.push(format!("seed {seed}: {error}"));
        } else {
            semantic_class_checks += 1;
            strict_class_checks += 1;
        }
    }

    let base = TurnAction {
        replace_three_of_a_kind: false,
        wildlife_wipes: vec![
            WildlifeWipe {
                slots: vec![MarketSlot::new(0).unwrap()],
            },
            WildlifeWipe {
                slots: vec![MarketSlot::new(1).unwrap()],
            },
        ],
        draft: DraftChoice::Paired {
            slot: MarketSlot::new(2).unwrap(),
        },
        tile: TilePlacement {
            coord: cascadia_game::HexCoord::new(0, 0),
            rotation: cascadia_game::Rotation::ZERO,
        },
        wildlife: None,
    };
    let mut reversed = base.clone();
    reversed.wildlife_wipes.reverse();
    let ordered_trace_rejections =
        usize::from(hidden_effect_trace(&base) != hidden_effect_trace(&reversed));
    if ordered_trace_rejections != 1 {
        failures.push("ordered paid-wipe traces were not distinguished".to_owned());
    }

    let scientific = AdversarialScientific {
        schema_version: SCHEMA_VERSION,
        experiment_id: EXPERIMENT_ID.to_owned(),
        protocol_id: PROTOCOL_ID.to_owned(),
        synthetic_states: 3,
        exhaustive_legal_actions,
        positive_duplicate_witnesses,
        ordered_trace_rejections,
        semantic_class_checks,
        strict_class_checks,
        passed: failures.is_empty()
            && positive_duplicate_witnesses == 3
            && ordered_trace_rejections == 1,
        failures,
    };
    Ok(AdversarialReport {
        report_id: report_id(&scientific)?,
        scientific,
    })
}

pub fn run_duplicate_accounting_smoke(
    dataset_root: &Path,
) -> Result<DuplicateSmokeReport, Box<dyn Error>> {
    let manifest_path = dataset_root.join("dataset.json");
    let manifest_bytes = fs::read(&manifest_path)?;
    let manifest: GradedOracleDatasetManifest = serde_json::from_slice(&manifest_bytes)?;
    validate_graded_oracle_dataset(dataset_root, &manifest)?;
    if !matches!(
        manifest.split,
        DatasetSplit::Train | DatasetSplit::Validation
    ) {
        return Err("S7 duplicate smoke refuses sealed data".into());
    }
    if manifest.game != GameConfig::research_aaaaa(4)? {
        return Err("S7 duplicate smoke requires the frozen four-player AAAAA ruleset".into());
    }
    let shard = manifest
        .shards
        .first()
        .ok_or("S7 duplicate smoke dataset has no shard")?;
    let groups = read_graded_oracle_shard(dataset_root, manifest.split, shard)?;
    let source_group = groups
        .first()
        .ok_or("S7 duplicate smoke shard has no group")?;
    let game = GameState::new(
        GameConfig::research_aaaaa(4)?,
        GameSeed::from_u64(shard.first_game_index),
    )?;
    let mut checks = CensusChecks::default();
    verify_group_parent(&game, source_group, &mut checks)?;

    let duplicate_index = usize::from(source_group.selected_index);
    let duplicate = source_group
        .candidates
        .get(duplicate_index)
        .ok_or("S7 duplicate smoke selected index is invalid")?
        .clone();
    let mut group = source_group.clone();
    let original_candidates = group.candidates.len();
    group.candidates.push(duplicate);
    let record = analyze_group(manifest.split.id(), 0, &game, &group, &mut checks)?;
    checks.selected_groups = 1;
    checks.selected_candidates = record.candidates;
    let passed = record.candidates == original_candidates + 1
        && record.selected_safe_class_size >= 2
        && record.serving_safe.duplicate_classes >= 1
        && record.serving_safe.collapsed_candidates >= 1
        && record.exact_public_within_safe.duplicate_classes >= 1
        && record.exact_public_within_safe.collapsed_candidates >= 1
        && record
            .exact_hidden_successor_within_safe
            .collapsed_candidates
            >= 1
        && checks.semantic_successor_parity_classes >= 1
        && checks.strict_successor_parity_classes >= 1
        && checks.invariant_failures == 0;
    if !passed {
        return Err("S7 duplicate smoke did not exercise exact production accounting".into());
    }
    let scientific = DuplicateSmokeScientific {
        schema_version: SCHEMA_VERSION,
        experiment_id: EXPERIMENT_ID.to_owned(),
        protocol_id: PROTOCOL_ID.to_owned(),
        dataset_id: manifest.dataset_id,
        manifest_blake3: blake3::hash(&manifest_bytes).to_hex().to_string(),
        split: manifest.split.id().to_owned(),
        group_id: source_group.group_id,
        original_candidates,
        duplicated_candidate_index: duplicate_index,
        checks,
        record,
        passed,
    };
    Ok(DuplicateSmokeReport {
        report_id: report_id(&scientific)?,
        scientific,
    })
}

fn validate_adversarial_state(
    game: &GameState,
    trunk: &PublicStateTrunk,
    rows: &[(TurnAction, ActionEdit)],
) -> Result<(), Box<dyn Error>> {
    let prepared = trunk.prepare_action_edits()?;
    let mut groups = BTreeMap::<Vec<u8>, Vec<(TurnAction, Vec<u8>, Vec<u8>)>>::new();
    for (action, edit) in rows {
        let applied = prepared.apply(edit)?;
        let mut record = applied.record;
        record.game_index = 0;
        record.targets.fill(0);
        let supply = applied.supply.canonical_bytes()?;
        let trace = hidden_effect_trace(action);
        let key = framed_material(
            SAFE_KEY_DOMAIN,
            [
                record.to_bytes().as_slice(),
                supply.as_slice(),
                trace.as_slice(),
            ],
        );
        groups
            .entry(key)
            .or_default()
            .push((action.clone(), record.to_bytes().to_vec(), supply));
    }
    for members in groups.values().filter(|members| members.len() > 1) {
        let mut exact_public = None;
        let mut exact_successor = None;
        for (action, expected_record, expected_supply) in members {
            let public = game.preview_public_afterstate(action)?;
            let mut record =
                PositionRecord::observe_public_for_seat(&public, 0, game.current_player());
            record.targets.fill(0);
            if record.to_bytes().as_slice() != expected_record {
                return Err("synthetic R3/public parity failed".into());
            }
            let supply =
                SupplySnapshot::from_exact(&ExactSemanticSupply::from_public_state(&public)?)
                    .canonical_bytes()?;
            if &supply != expected_supply {
                return Err("synthetic semantic supply parity failed".into());
            }
            let public_bytes = public.canonical_bytes();
            let successor_bytes = game.transition(action)?.canonical_bytes();
            if let Some(expected) = &exact_public {
                if expected != &public_bytes {
                    return Err("positive duplicate changed exact public state".into());
                }
            } else {
                exact_public = Some(public_bytes);
            }
            if let Some(expected) = &exact_successor {
                if expected != &successor_bytes {
                    return Err("positive duplicate changed exact hidden successor".into());
                }
            } else {
                exact_successor = Some(successor_bytes);
            }
        }
    }
    Ok(())
}

pub fn aggregate_reports_with_order_proof(
    reports: &[CensusReport],
    adversarial: &AdversarialReport,
) -> Result<(AggregateReport, AggregateReport, OrderProof), Box<dyn Error>> {
    if reports.is_empty() {
        return Err("S7 aggregate requires at least one shard".into());
    }
    if adversarial.scientific.experiment_id != EXPERIMENT_ID
        || adversarial.scientific.protocol_id != PROTOCOL_ID
        || adversarial.report_id != report_id(&adversarial.scientific)?
    {
        return Err("S7 adversarial report identity is invalid".into());
    }
    let forward = aggregate_reports(reports, adversarial)?;
    let mut reversed = reports.to_vec();
    reversed.reverse();
    let reverse = aggregate_reports(&reversed, adversarial)?;
    let forward_bytes = serde_json::to_vec(&forward.scientific)?;
    let reverse_bytes = serde_json::to_vec(&reverse.scientific)?;
    let scientific = OrderProofScientific {
        schema_version: SCHEMA_VERSION,
        experiment_id: EXPERIMENT_ID.to_owned(),
        protocol_id: PROTOCOL_ID.to_owned(),
        forward_report_id: forward.report_id.clone(),
        reverse_report_id: reverse.report_id.clone(),
        forward_scientific_blake3: blake3::hash(&forward_bytes).to_hex().to_string(),
        reverse_scientific_blake3: blake3::hash(&reverse_bytes).to_hex().to_string(),
        byte_identical: forward_bytes == reverse_bytes,
    };
    let proof = OrderProof {
        report_id: report_id(&scientific)?,
        scientific,
    };
    Ok((forward, reverse, proof))
}

fn aggregate_reports(
    reports: &[CensusReport],
    adversarial: &AdversarialReport,
) -> Result<AggregateReport, Box<dyn Error>> {
    let first = &reports[0].scientific;
    let shard_count = first.shard_count;
    let mut shard_indices = BTreeSet::new();
    let mut dataset_map = BTreeMap::<String, DatasetIdentity>::new();
    let mut records = Vec::new();
    let mut checks = CensusChecks::default();
    for report in reports {
        if report.report_id != report_id(&report.scientific)?
            || report.scientific.schema_version != SCHEMA_VERSION
            || report.scientific.experiment_id != EXPERIMENT_ID
            || report.scientific.protocol_id != PROTOCOL_ID
            || report.scientific.source_bundle_blake3 != first.source_bundle_blake3
            || report.scientific.executable_blake3 != first.executable_blake3
            || report.scientific.shard_count != shard_count
            || !report.scientific.complete_open_shard
        {
            return Err("S7 shard report identity or completeness is invalid".into());
        }
        if !shard_indices.insert(report.scientific.shard_index) {
            return Err("S7 shard index is duplicated".into());
        }
        for dataset in &report.scientific.datasets {
            match dataset_map.get(&dataset.split) {
                Some(existing) if existing != dataset => {
                    return Err(format!("S7 {} dataset identity disagrees", dataset.split).into());
                }
                _ => {
                    dataset_map.insert(dataset.split.clone(), dataset.clone());
                }
            }
        }
        checks.add_assign(&report.scientific.checks);
        records.extend(report.scientific.records.clone());
    }
    let expected_indices = (0..shard_count).collect::<BTreeSet<_>>();
    if shard_indices != expected_indices {
        return Err("S7 aggregate does not contain every shard index".into());
    }
    let datasets = dataset_map.into_values().collect::<Vec<_>>();
    if datasets.len() != 2
        || datasets
            .iter()
            .map(|dataset| dataset.split.as_str())
            .collect::<BTreeSet<_>>()
            != BTreeSet::from(["train", "validation"])
    {
        return Err("S7 aggregate requires exactly open train and validation data".into());
    }
    records.sort_by(|left, right| {
        (split_order(&left.split), left.row).cmp(&(split_order(&right.split), right.row))
    });
    validate_aggregate_coverage(&datasets, &records)?;

    let train = records
        .iter()
        .filter(|record| record.split == "train")
        .cloned()
        .collect::<Vec<_>>();
    let validation = records
        .iter()
        .filter(|record| record.split == "validation")
        .cloned()
        .collect::<Vec<_>>();
    let train_semantic_state_supply =
        reduction_distribution(&train, |record| &record.semantic_state_supply);
    let train_serving_safe = reduction_distribution(&train, |record| &record.serving_safe);
    let train_exact_public_within_safe =
        reduction_distribution(&train, |record| &record.exact_public_within_safe);
    let validation_semantic_state_supply =
        reduction_distribution(&validation, |record| &record.semantic_state_supply);
    let validation_serving_safe =
        reduction_distribution(&validation, |record| &record.serving_safe);
    let validation_exact_public_within_safe =
        reduction_distribution(&validation, |record| &record.exact_public_within_safe);
    let validation_trace_rejected_collapses = validation
        .iter()
        .map(|record| record.trace_rejected_collapses)
        .sum();
    let validation_semantic_identity_collapses_beyond_exact_public = validation
        .iter()
        .map(|record| record.semantic_identity_collapses_beyond_exact_public)
        .sum();

    let complete_open_corpus = records.len()
        == datasets.iter().map(|dataset| dataset.groups).sum::<usize>()
        && checks.selected_groups == records.len()
        && checks.selected_candidates
            == records
                .iter()
                .map(|record| record.candidates)
                .sum::<usize>();
    let semantic_duplicate_classes = records
        .iter()
        .map(|record| record.serving_safe.duplicate_classes)
        .sum::<usize>();
    let strict_duplicate_classes = records
        .iter()
        .map(|record| record.exact_public_within_safe.duplicate_classes)
        .sum::<usize>();
    let semantic_successor_parity =
        checks.semantic_successor_parity_classes == semantic_duplicate_classes;
    let strict_successor_parity =
        checks.strict_successor_parity_classes == strict_duplicate_classes;
    let validation_median_reduction_pass =
        validation_serving_safe.median_reduction_ppm >= PROMOTION_MEDIAN_PPM;
    let validation_tail_reduction_pass = validation_serving_safe.p90_reduction_ppm
        >= PROMOTION_TAIL_PPM
        && validation_serving_safe.p90_collapsed_candidates >= PROMOTION_TAIL_ABSOLUTE;
    let gates = PromotionGates {
        adversarial_pass: adversarial.scientific.passed,
        complete_open_corpus,
        zero_invariant_failures: checks.invariant_failures == 0,
        semantic_successor_parity,
        strict_successor_parity,
        validation_median_reduction_pass,
        validation_tail_reduction_pass,
        promotion_pass: adversarial.scientific.passed
            && complete_open_corpus
            && checks.invariant_failures == 0
            && semantic_successor_parity
            && strict_successor_parity
            && (validation_median_reduction_pass || validation_tail_reduction_pass),
    };
    let valid = gates.adversarial_pass
        && gates.complete_open_corpus
        && gates.zero_invariant_failures
        && gates.semantic_successor_parity
        && gates.strict_successor_parity;
    let classification = if !valid {
        CLASSIFICATION_INVALID
    } else if gates.promotion_pass {
        CLASSIFICATION_PROMISING
    } else {
        CLASSIFICATION_FUTILE
    };
    let scientific = AggregateScientific {
        schema_version: SCHEMA_VERSION,
        experiment_id: EXPERIMENT_ID.to_owned(),
        protocol_id: PROTOCOL_ID.to_owned(),
        source_bundle_blake3: first.source_bundle_blake3.clone(),
        executable_blake3: first.executable_blake3.clone(),
        valid,
        classification: classification.to_owned(),
        datasets,
        checks,
        train_semantic_state_supply,
        train_serving_safe,
        train_exact_public_within_safe,
        validation_semantic_state_supply,
        validation_serving_safe,
        validation_exact_public_within_safe,
        validation_trace_rejected_collapses,
        validation_semantic_identity_collapses_beyond_exact_public,
        gates,
        records,
    };
    Ok(AggregateReport {
        report_id: report_id(&scientific)?,
        scientific,
    })
}

fn validate_aggregate_coverage(
    datasets: &[DatasetIdentity],
    records: &[GroupRecord],
) -> Result<(), Box<dyn Error>> {
    let mut cursor = 0usize;
    for dataset in datasets {
        let mut rows = records
            .iter()
            .filter(|record| record.split == dataset.split)
            .map(|record| record.row)
            .collect::<Vec<_>>();
        rows.sort_unstable();
        if rows != (0..dataset.groups).collect::<Vec<_>>() {
            return Err(format!("S7 {} rows are incomplete or duplicated", dataset.split).into());
        }
        cursor += rows.len();
    }
    if cursor != records.len() {
        return Err("S7 aggregate contains a record for an unknown split".into());
    }
    Ok(())
}

fn reduction_distribution(
    records: &[GroupRecord],
    tier: impl Fn(&GroupRecord) -> &ClassStats,
) -> ReductionDistribution {
    let candidates = records.iter().map(|record| tier(record).candidates).sum();
    let unique_classes = records
        .iter()
        .map(|record| tier(record).unique_classes)
        .sum();
    let collapsed_candidates = candidates - unique_classes;
    let mut reductions = records
        .iter()
        .map(|record| u64::try_from(tier(record).collapsed_candidates).unwrap())
        .collect::<Vec<_>>();
    let mut fractions = records
        .iter()
        .map(|record| tier(record).reduction_ppm)
        .collect::<Vec<_>>();
    reductions.sort_unstable();
    fractions.sort_unstable();
    ReductionDistribution {
        groups: records.len(),
        candidates,
        unique_classes,
        collapsed_candidates,
        weighted_reduction_ppm: ratio_ppm(collapsed_candidates as u64, candidates as u64),
        median_reduction_ppm: nearest_rank(&fractions, 50),
        p90_reduction_ppm: nearest_rank(&fractions, 90),
        p99_reduction_ppm: nearest_rank(&fractions, 99),
        maximum_reduction_ppm: fractions.last().copied().unwrap_or(0),
        median_collapsed_candidates: nearest_rank(&reductions, 50),
        p90_collapsed_candidates: nearest_rank(&reductions, 90),
        p99_collapsed_candidates: nearest_rank(&reductions, 99),
        maximum_collapsed_candidates: reductions.last().copied().unwrap_or(0),
        groups_with_any_collapse: records
            .iter()
            .filter(|record| tier(record).collapsed_candidates > 0)
            .count(),
    }
}

fn nearest_rank(sorted: &[u64], percentile: u64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let rank = ((sorted.len() as u64 * percentile).div_ceil(100)).max(1) as usize;
    sorted[rank - 1]
}

fn selected_row_count(total: usize, shard_index: usize, shard_count: usize) -> usize {
    (0..total)
        .filter(|row| row % shard_count == shard_index)
        .count()
}

fn split_order(split: &str) -> u8 {
    match split {
        "train" => 0,
        "validation" => 1,
        _ => 2,
    }
}

fn ratio_ppm(numerator: u64, denominator: u64) -> u64 {
    if denominator == 0 {
        0
    } else {
        ((u128::from(numerator) * 1_000_000 + u128::from(denominator / 2))
            / u128::from(denominator)) as u64
    }
}

fn report_id(value: &impl Serialize) -> Result<String, Box<dyn Error>> {
    let mut hasher = Hasher::new();
    hasher.update(REPORT_DOMAIN);
    hasher.update(&serde_json::to_vec(value)?);
    Ok(hasher.finalize().to_hex().to_string())
}

fn validate_lower_hex_digest(
    label: &str,
    value: &str,
    expected_len: usize,
) -> Result<(), Box<dyn Error>> {
    if value.len() != expected_len
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(format!("{label} digest must be {expected_len} lowercase hex bytes").into());
    }
    Ok(())
}

fn checksum_file(path: &Path) -> Result<String, Box<dyn Error>> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut hasher = Hasher::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hasher.finalize().to_hex().to_string())
}

pub fn read_json<T: serde::de::DeserializeOwned>(path: &PathBuf) -> Result<T, Box<dyn Error>> {
    Ok(serde_json::from_reader(BufReader::new(File::open(path)?))?)
}

pub fn write_json_atomic(path: &Path, value: &impl Serialize) -> Result<(), Box<dyn Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temp = path.with_extension(format!(
        "{}.tmp",
        path.extension()
            .and_then(|extension| extension.to_str())
            .unwrap_or("json")
    ));
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    {
        let mut file = File::create(&temp)?;
        file.write_all(&bytes)?;
        file.sync_all()?;
    }
    fs::rename(temp, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn class_map(sizes: &[usize]) -> BTreeMap<Vec<u8>, Vec<usize>> {
        let mut next = 0usize;
        sizes
            .iter()
            .enumerate()
            .map(|(class, size)| {
                let members = (next..next + size).collect::<Vec<_>>();
                next += size;
                (vec![class as u8], members)
            })
            .collect()
    }

    #[test]
    fn ordered_wipes_are_not_equivalent() {
        let slot0 = MarketSlot::new(0).unwrap();
        let slot1 = MarketSlot::new(1).unwrap();
        let base = TurnAction {
            replace_three_of_a_kind: false,
            wildlife_wipes: vec![
                WildlifeWipe { slots: vec![slot0] },
                WildlifeWipe { slots: vec![slot1] },
            ],
            draft: DraftChoice::Paired { slot: slot0 },
            tile: TilePlacement {
                coord: cascadia_game::HexCoord::new(0, 0),
                rotation: cascadia_game::Rotation::ZERO,
            },
            wildlife: None,
        };
        let mut reversed = base.clone();
        reversed.wildlife_wipes.reverse();
        assert_ne!(hidden_effect_trace(&base), hidden_effect_trace(&reversed));
    }

    #[test]
    fn class_accounting_is_exact() {
        let map = class_map(&[1, 2, 4]);
        let stats = class_stats_from_map(7, &map);
        assert_eq!(stats.unique_classes, 3);
        assert_eq!(stats.collapsed_candidates, 4);
        assert_eq!(stats.duplicate_classes, 2);
        assert_eq!(stats.candidates_in_duplicate_classes, 6);
        assert_eq!(stats.pair_collisions, 7);
        assert_eq!(stats.maximum_class_size, 4);
    }

    #[test]
    fn semantic_collapses_beyond_exact_public_count_subclasses() {
        assert_eq!(collapses_beyond_exact_public(4, 1).unwrap(), 0);
        assert_eq!(collapses_beyond_exact_public(4, 2).unwrap(), 1);
        assert_eq!(collapses_beyond_exact_public(4, 4).unwrap(), 3);
        assert!(collapses_beyond_exact_public(1, 1).is_err());
        assert!(collapses_beyond_exact_public(3, 0).is_err());
        assert!(collapses_beyond_exact_public(3, 4).is_err());
    }

    #[test]
    fn reduction_quantiles_use_nearest_rank() {
        assert_eq!(nearest_rank(&[0, 10, 20, 30], 50), 10);
        assert_eq!(nearest_rank(&[0, 10, 20, 30], 90), 30);
        assert_eq!(nearest_rank(&[], 99), 0);
    }

    #[test]
    fn source_digest_validation_fails_closed() {
        assert!(validate_lower_hex_digest("source", &"a".repeat(64), 64).is_ok());
        assert!(validate_lower_hex_digest("source", &"A".repeat(64), 64).is_err());
        assert!(validate_lower_hex_digest("source", &"a".repeat(63), 64).is_err());
    }

    #[test]
    fn framed_material_is_unambiguous() {
        assert_ne!(
            framed_material(b"x", [b"ab".as_slice(), b"c".as_slice()]),
            framed_material(b"x", [b"a".as_slice(), b"bc".as_slice()])
        );
    }
}
