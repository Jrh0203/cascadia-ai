use std::{
    collections::{BTreeMap, BTreeSet},
    env,
    error::Error,
    ffi::OsString,
    fs::{self, File},
    io::{BufReader, Write},
    path::{Path, PathBuf},
    process::Command,
};

use blake3::Hasher;
use cascadia_data::{
    DatasetSplit, OpponentIntentDatasetManifest, OpponentIntentRecord, PublicActionRecord,
    read_opponent_intent_shard_records, validate_opponent_intent_dataset,
};
use cascadia_provenance::{SourceProvenance, checksum_file, source_provenance};
use cascadia_sim::StrategyKind;
use serde::Serialize;

const SCHEMA_VERSION: u16 = 1;
const EXPERIMENT_ID: &str = "o1-opponent-intent-policy-heldout-corpus-v1";
const EXPECTED_GAMES: usize = 1_664;
const EXPECTED_RECORDS: usize = 126_464;
const POLICY_COUNT: usize = 6;
const HELP: &str = concat!(
    "Usage: opponent_intent_policy_corpus_audit \\\n",
    "  --dataset train-part-0=PATH \\\n",
    "  --dataset train-part-1=PATH \\\n",
    "  --dataset validation=PATH \\\n",
    "  --dataset test=PATH \\\n",
    "  --dataset final-stress=PATH \\\n",
    "  --output PATH\n\n",
    "Validates the frozen O1 policy-held-out corpus, proves model-input identity\n",
    "exclusion, measures policy/action/survival support, and rejects exact model-\n",
    "input overlap between every corpus pair.\n",
);

#[derive(Debug, Clone, PartialEq, Eq)]
struct Args {
    datasets: BTreeMap<DatasetRole, PathBuf>,
    output: PathBuf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
enum DatasetRole {
    #[serde(rename = "train-part-0")]
    TrainPart0,
    #[serde(rename = "train-part-1")]
    TrainPart1,
    #[serde(rename = "validation")]
    Validation,
    #[serde(rename = "test")]
    Test,
    #[serde(rename = "final-stress")]
    FinalStress,
}

impl DatasetRole {
    const ALL: [Self; 5] = [
        Self::TrainPart0,
        Self::TrainPart1,
        Self::Validation,
        Self::Test,
        Self::FinalStress,
    ];

    const fn id(self) -> &'static str {
        match self {
            Self::TrainPart0 => "train-part-0",
            Self::TrainPart1 => "train-part-1",
            Self::Validation => "validation",
            Self::Test => "test",
            Self::FinalStress => "final-stress",
        }
    }

    fn parse(value: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|role| role.id() == value)
    }
}

#[derive(Debug, Clone)]
struct ExpectedDataset {
    split: DatasetSplit,
    first_game_index: u64,
    games: usize,
    cohort_id: &'static str,
    policy_pool: Vec<StrategyKind>,
    required_policy: Option<StrategyKind>,
}

#[derive(Debug, Serialize)]
struct AuditReport {
    schema_version: u16,
    experiment_id: &'static str,
    status: &'static str,
    classification: &'static str,
    scientific: ScientificReport,
    scientific_blake3: String,
    provenance: ExecutionProvenance,
}

#[derive(Debug, Clone, Serialize)]
struct ScientificReport {
    datasets: Vec<DatasetAudit>,
    totals: CorpusTotals,
    overlaps: Vec<ModelInputOverlap>,
    action_factor_coverage: Vec<FactorCoverage>,
    survival_coverage: SurvivalCoverage,
    limitations: Vec<ScopeLimitation>,
    gates: Vec<Gate>,
}

#[derive(Debug, Clone, Default, Serialize)]
struct CorpusTotals {
    games: usize,
    records: usize,
    shards: usize,
    unique_model_inputs: usize,
    duplicate_model_inputs_within_datasets: usize,
    identity_exclusion_checks: usize,
}

#[derive(Debug, Clone, Serialize)]
struct DatasetAudit {
    role: DatasetRole,
    dataset_id: String,
    manifest_blake3: String,
    split: String,
    first_game_index: u64,
    games: usize,
    records: usize,
    shards: usize,
    cohort_id: String,
    policy_pool: Vec<String>,
    required_policy: Option<String>,
    seat_policy_counts: [u64; POLICY_COUNT],
    target_policy_counts: [u64; POLICY_COUNT],
    model_input_bytes: usize,
    unique_model_inputs: usize,
    duplicate_model_inputs: usize,
    model_input_set_blake3: String,
    identity_exclusion_checks: usize,
    target_action_support: ActionSupport,
    history_action_support: ActionSupport,
    survival_support: SurvivalSupport,
}

#[derive(Debug)]
struct AuditedDataset {
    report: DatasetAudit,
    model_input_hashes: BTreeSet<[u8; 32]>,
}

#[derive(Debug, Clone, Default, Serialize)]
struct ActionSupport {
    actions: u64,
    draft_kind: [u64; 2],
    tile_slot: [u64; 4],
    wildlife_slot: [u64; 4],
    rotation: [u64; 6],
    drafted_wildlife: [u64; 5],
    wildlife_present: [u64; 2],
    replace_three_of_a_kind: [u64; 2],
    paid_wipe_count: BTreeMap<u8, u64>,
    paid_wipe_total_slots: BTreeMap<u8, u64>,
}

impl ActionSupport {
    fn observe(&mut self, action: PublicActionRecord) -> Result<(), Box<dyn Error>> {
        let draft_kind = checked_index(action.draft_kind, 2, "draft kind")?;
        let tile_slot = checked_index(action.tile_slot, 4, "tile slot")?;
        let wildlife_slot = checked_index(action.wildlife_slot, 4, "wildlife slot")?;
        let rotation = checked_index(action.rotation, 6, "rotation")?;
        let wildlife = checked_index(action.drafted_wildlife, 5, "drafted wildlife species")?;
        let wildlife_present =
            checked_index(action.wildlife_present, 2, "wildlife placement flag")?;
        let replacement = checked_index(
            action.replace_three_of_a_kind,
            2,
            "three-of-a-kind replacement flag",
        )?;
        self.actions += 1;
        self.draft_kind[draft_kind] += 1;
        self.tile_slot[tile_slot] += 1;
        self.wildlife_slot[wildlife_slot] += 1;
        self.rotation[rotation] += 1;
        self.drafted_wildlife[wildlife] += 1;
        self.wildlife_present[wildlife_present] += 1;
        self.replace_three_of_a_kind[replacement] += 1;
        *self
            .paid_wipe_count
            .entry(action.paid_wipe_count)
            .or_default() += 1;
        *self
            .paid_wipe_total_slots
            .entry(action.paid_wipe_total_slots)
            .or_default() += 1;
        Ok(())
    }

    fn merge(&mut self, other: &Self) {
        self.actions += other.actions;
        merge_array(&mut self.draft_kind, &other.draft_kind);
        merge_array(&mut self.tile_slot, &other.tile_slot);
        merge_array(&mut self.wildlife_slot, &other.wildlife_slot);
        merge_array(&mut self.rotation, &other.rotation);
        merge_array(&mut self.drafted_wildlife, &other.drafted_wildlife);
        merge_array(&mut self.wildlife_present, &other.wildlife_present);
        merge_array(
            &mut self.replace_three_of_a_kind,
            &other.replace_three_of_a_kind,
        );
        merge_map(&mut self.paid_wipe_count, &other.paid_wipe_count);
        merge_map(
            &mut self.paid_wipe_total_slots,
            &other.paid_wipe_total_slots,
        );
    }

    fn factor_classes(&self) -> BTreeMap<&'static str, BTreeSet<u8>> {
        BTreeMap::from([
            ("draft_kind", populated_indices(&self.draft_kind)),
            ("tile_slot", populated_indices(&self.tile_slot)),
            ("wildlife_slot", populated_indices(&self.wildlife_slot)),
            ("rotation", populated_indices(&self.rotation)),
            (
                "drafted_wildlife",
                populated_indices(&self.drafted_wildlife),
            ),
            (
                "wildlife_present",
                populated_indices(&self.wildlife_present),
            ),
            (
                "replace_three_of_a_kind",
                populated_indices(&self.replace_three_of_a_kind),
            ),
            (
                "paid_wipe_count",
                self.paid_wipe_count
                    .iter()
                    .filter_map(|(class, count)| (*count != 0).then_some(*class))
                    .collect(),
            ),
            (
                "paid_wipe_total_slots",
                self.paid_wipe_total_slots
                    .iter()
                    .filter_map(|(class, count)| (*count != 0).then_some(*class))
                    .collect(),
            ),
        ])
    }
}

#[derive(Debug, Clone, Default, Serialize)]
struct SurvivalSupport {
    targets: u64,
    disposition: [u64; 4],
    pair_survives: [u64; 2],
    final_slot_when_survived: [u64; 4],
    initial_wildlife: [u64; 5],
}

impl SurvivalSupport {
    fn observe(&mut self, target: cascadia_data::TileSurvivalTarget) -> Result<(), Box<dyn Error>> {
        let disposition = usize::from(
            target
                .disposition
                .checked_sub(1)
                .ok_or("survival disposition is zero")?,
        );
        if disposition >= self.disposition.len() {
            return Err("survival disposition is out of range".into());
        }
        let pair = checked_index(target.pair_survives, 2, "pair survival flag")?;
        let wildlife = checked_index(target.initial_wildlife, 5, "initial wildlife")?;
        self.targets += 1;
        self.disposition[disposition] += 1;
        self.pair_survives[pair] += 1;
        self.initial_wildlife[wildlife] += 1;
        if target.disposition == 4 {
            let slot = checked_index(target.final_slot, 4, "surviving final slot")?;
            self.final_slot_when_survived[slot] += 1;
        }
        Ok(())
    }

    fn merge(&mut self, other: &Self) {
        self.targets += other.targets;
        merge_array(&mut self.disposition, &other.disposition);
        merge_array(&mut self.pair_survives, &other.pair_survives);
        merge_array(
            &mut self.final_slot_when_survived,
            &other.final_slot_when_survived,
        );
        merge_array(&mut self.initial_wildlife, &other.initial_wildlife);
    }
}

#[derive(Debug, Clone, Serialize)]
struct ModelInputOverlap {
    left: DatasetRole,
    right: DatasetRole,
    exact_hash_overlap: usize,
    sample_hashes: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct FactorCoverage {
    factor: String,
    training_classes: Vec<u8>,
    held_out_evaluation_classes: Vec<u8>,
    missing_from_training: Vec<u8>,
    passed: bool,
}

#[derive(Debug, Clone, Serialize)]
struct SurvivalCoverage {
    training_dispositions: Vec<u8>,
    held_out_evaluation_dispositions: Vec<u8>,
    training_pair_survival_classes: Vec<u8>,
    held_out_evaluation_pair_survival_classes: Vec<u8>,
    missing_dispositions: Vec<u8>,
    missing_pair_survival_classes: Vec<u8>,
    passed: bool,
}

#[derive(Debug, Clone, Serialize)]
struct Gate {
    label: &'static str,
    passed: bool,
    observed: String,
}

#[derive(Debug, Clone, Serialize)]
struct ScopeLimitation {
    label: &'static str,
    observed: String,
    consequence: &'static str,
}

#[derive(Debug, Serialize)]
struct ExecutionProvenance {
    source: SourceProvenance,
    executable_blake3: String,
    hostname: String,
    logical_parallelism: usize,
    dataset_roots: BTreeMap<&'static str, String>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args(env::args_os().skip(1))?;
    let mut audited = Vec::with_capacity(DatasetRole::ALL.len());
    for role in DatasetRole::ALL {
        audited.push(audit_dataset(
            role,
            args.datasets
                .get(&role)
                .expect("argument parser requires every dataset role"),
        )?);
    }

    let totals = corpus_totals(&audited);
    let overlaps = model_input_overlaps(&audited);
    let training_actions = combined_action_support(
        &audited,
        &[DatasetRole::TrainPart0, DatasetRole::TrainPart1],
    );
    let evaluation_actions =
        combined_action_support(&audited, &[DatasetRole::Validation, DatasetRole::Test]);
    let action_factor_coverage = action_factor_coverage(&training_actions, &evaluation_actions);
    let training_survival = combined_survival_support(
        &audited,
        &[DatasetRole::TrainPart0, DatasetRole::TrainPart1],
    );
    let evaluation_survival =
        combined_survival_support(&audited, &[DatasetRole::Validation, DatasetRole::Test]);
    let survival_coverage = survival_coverage(&training_survival, &evaluation_survival);
    let limitations = scope_limitations(&audited);
    let gates = gates(
        &audited,
        &totals,
        &overlaps,
        &action_factor_coverage,
        &survival_coverage,
    );
    let passed = gates.iter().all(|gate| gate.passed);
    let scientific = ScientificReport {
        datasets: audited
            .iter()
            .map(|dataset| dataset.report.clone())
            .collect(),
        totals,
        overlaps,
        action_factor_coverage,
        survival_coverage,
        limitations,
        gates,
    };
    let scientific_blake3 = blake3::hash(&serde_json::to_vec(&scientific)?)
        .to_hex()
        .to_string();
    let report = AuditReport {
        schema_version: SCHEMA_VERSION,
        experiment_id: EXPERIMENT_ID,
        status: "complete",
        classification: if passed {
            "policy_held_out_corpus_passed"
        } else {
            "policy_held_out_corpus_failed"
        },
        scientific,
        scientific_blake3,
        provenance: ExecutionProvenance {
            source: source_provenance()?,
            executable_blake3: checksum_file(&env::current_exe()?)?,
            hostname: command_output("hostname", &[]).unwrap_or_else(|| "unknown".to_owned()),
            logical_parallelism: std::thread::available_parallelism()
                .map(usize::from)
                .unwrap_or(1),
            dataset_roots: DatasetRole::ALL
                .into_iter()
                .map(|role| (role.id(), args.datasets[&role].display().to_string()))
                .collect(),
        },
    };
    write_json_atomically(&args.output, &report)?;
    println!(
        "{}",
        serde_json::json!({
            "classification": report.classification,
            "datasets": report.scientific.datasets.len(),
            "games": report.scientific.totals.games,
            "records": report.scientific.totals.records,
            "scientific_blake3": report.scientific_blake3,
            "output": args.output,
        })
    );
    Ok(())
}

fn parse_args(
    arguments: impl IntoIterator<Item = impl Into<OsString>>,
) -> Result<Args, Box<dyn Error>> {
    let mut arguments = arguments
        .into_iter()
        .map(Into::into)
        .collect::<Vec<_>>()
        .into_iter();
    let mut datasets = BTreeMap::new();
    let mut output = None;
    while let Some(argument) = arguments.next() {
        let argument = argument
            .to_str()
            .ok_or("command-line arguments must be valid UTF-8")?;
        match argument {
            "--dataset" => {
                let value = arguments
                    .next()
                    .ok_or("--dataset requires ROLE=PATH")?
                    .into_string()
                    .map_err(|_| "dataset argument must be valid UTF-8")?;
                let (role, path) = value
                    .split_once('=')
                    .ok_or("--dataset requires ROLE=PATH")?;
                let role = DatasetRole::parse(role)
                    .ok_or_else(|| format!("unknown dataset role {role}"))?;
                if path.is_empty() || datasets.insert(role, PathBuf::from(path)).is_some() {
                    return Err(format!("invalid or duplicate dataset role {}", role.id()).into());
                }
            }
            "--output" => {
                output = Some(PathBuf::from(
                    arguments.next().ok_or("--output requires a path")?,
                ));
            }
            "--help" | "-h" => {
                println!("{HELP}");
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument {other}\n\n{HELP}").into()),
        }
    }
    let missing = DatasetRole::ALL
        .into_iter()
        .filter(|role| !datasets.contains_key(role))
        .map(DatasetRole::id)
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(format!("missing dataset roles: {}", missing.join(", ")).into());
    }
    Ok(Args {
        datasets,
        output: output.ok_or("--output is required")?,
    })
}

fn expected_dataset(role: DatasetRole) -> ExpectedDataset {
    use StrategyKind::{
        Greedy, PatternAware, PatternCommitment, PatternCompetition, PatternPortfolio, Random,
    };
    match role {
        DatasetRole::TrainPart0 => ExpectedDataset {
            split: DatasetSplit::Train,
            first_game_index: 0,
            games: 512,
            cohort_id: "o1-train-mixed-v1",
            policy_pool: vec![Greedy, PatternAware, PatternCommitment],
            required_policy: None,
        },
        DatasetRole::TrainPart1 => ExpectedDataset {
            split: DatasetSplit::Train,
            first_game_index: 512,
            games: 512,
            cohort_id: "o1-train-mixed-v1",
            policy_pool: vec![Greedy, PatternAware, PatternCommitment],
            required_policy: None,
        },
        DatasetRole::Validation => ExpectedDataset {
            split: DatasetSplit::Validation,
            first_game_index: 100_000,
            games: 256,
            cohort_id: "o1-validation-heldout-competition-v1",
            policy_pool: vec![Greedy, PatternAware, PatternCommitment, PatternCompetition],
            required_policy: Some(PatternCompetition),
        },
        DatasetRole::Test => ExpectedDataset {
            split: DatasetSplit::Test,
            first_game_index: 200_000,
            games: 256,
            cohort_id: "o1-test-heldout-portfolio-v1",
            policy_pool: vec![
                Greedy,
                PatternAware,
                PatternCommitment,
                PatternCompetition,
                PatternPortfolio,
            ],
            required_policy: Some(PatternPortfolio),
        },
        DatasetRole::FinalStress => ExpectedDataset {
            split: DatasetSplit::Final,
            first_game_index: 300_000,
            games: 128,
            cohort_id: "o1-final-heldout-random-v1",
            policy_pool: vec![
                Random,
                Greedy,
                PatternAware,
                PatternCommitment,
                PatternCompetition,
                PatternPortfolio,
            ],
            required_policy: Some(Random),
        },
    }
}

fn audit_dataset(role: DatasetRole, root: &Path) -> Result<AuditedDataset, Box<dyn Error>> {
    let manifest_path = root.join("dataset.json");
    let manifest: OpponentIntentDatasetManifest =
        serde_json::from_reader(BufReader::new(File::open(&manifest_path)?))?;
    validate_expected_manifest(role, &manifest)?;
    validate_opponent_intent_dataset(root, &manifest)?;

    let mut model_input_hashes = BTreeSet::new();
    let mut duplicate_model_inputs = 0usize;
    let mut identity_exclusion_checks = 0usize;
    let mut seat_policy_counts = [0u64; POLICY_COUNT];
    let mut target_policy_counts = [0u64; POLICY_COUNT];
    let mut target_action_support = ActionSupport::default();
    let mut history_action_support = ActionSupport::default();
    let mut survival_support = SurvivalSupport::default();
    let mut model_input_bytes = None;
    for shard in &manifest.shards {
        for record in read_opponent_intent_shard_records(root, manifest.split, shard)? {
            let input = record.model_input_bytes();
            if let Some(expected) = model_input_bytes {
                if input.len() != expected {
                    return Err("model-input byte width is not fixed".into());
                }
            } else {
                model_input_bytes = Some(input.len());
            }
            let hash = *blake3::hash(&input).as_bytes();
            if !model_input_hashes.insert(hash) {
                duplicate_model_inputs += 1;
            }
            prove_identity_exclusion(&record, &input)?;
            identity_exclusion_checks += 1;
            if record.focal_turn == 0 {
                for code in record.seat_policy_codes {
                    seat_policy_counts[checked_index(code, POLICY_COUNT, "seat policy")?] += 1;
                }
            }
            for target in record.opponent_targets {
                target_policy_counts
                    [checked_index(target.policy_code, POLICY_COUNT, "target policy")?] += 1;
                target_action_support.observe(target.action)?;
            }
            for entry in record
                .history
                .iter()
                .take(usize::from(record.history_count))
            {
                history_action_support.observe(entry.action)?;
            }
            for target in record.survival_targets {
                survival_support.observe(target)?;
            }
        }
    }
    let model_input_set_blake3 = digest_hash_set(&model_input_hashes);
    Ok(AuditedDataset {
        report: DatasetAudit {
            role,
            dataset_id: manifest.dataset_id,
            manifest_blake3: checksum_file(&manifest_path)?,
            split: manifest.split.id().to_owned(),
            first_game_index: manifest.first_game_index,
            games: manifest.completed_games,
            records: manifest.total_records,
            shards: manifest.shards.len(),
            cohort_id: manifest.cohort.cohort_id,
            policy_pool: manifest
                .cohort
                .policy_pool
                .iter()
                .map(|policy| policy.id().to_owned())
                .collect(),
            required_policy: manifest
                .cohort
                .required_policy
                .map(|policy| policy.id().to_owned()),
            seat_policy_counts,
            target_policy_counts,
            model_input_bytes: model_input_bytes.unwrap_or(0),
            unique_model_inputs: model_input_hashes.len(),
            duplicate_model_inputs,
            model_input_set_blake3,
            identity_exclusion_checks,
            target_action_support,
            history_action_support,
            survival_support,
        },
        model_input_hashes,
    })
}

fn validate_expected_manifest(
    role: DatasetRole,
    manifest: &OpponentIntentDatasetManifest,
) -> Result<(), Box<dyn Error>> {
    let expected = expected_dataset(role);
    if manifest.split != expected.split
        || manifest.first_game_index != expected.first_game_index
        || manifest.requested_games != expected.games
        || manifest.completed_games != expected.games
        || manifest.total_records != expected.games * 76
        || manifest.cohort.cohort_id != expected.cohort_id
        || manifest.cohort.policy_pool != expected.policy_pool
        || manifest.cohort.required_policy != expected.required_policy
    {
        return Err(format!("{} manifest violates the frozen corpus contract", role.id()).into());
    }
    Ok(())
}

fn prove_identity_exclusion(
    record: &OpponentIntentRecord,
    expected_input: &[u8],
) -> Result<(), Box<dyn Error>> {
    let mut changed = record.clone();
    changed.game_index ^= u64::MAX;
    changed.position.game_index ^= u64::MAX;
    for target in &mut changed.position.targets {
        *target = target.wrapping_add(1);
    }
    changed.seat_policy_codes = [5, 4, 3, 2];
    for target in &mut changed.opponent_targets {
        target.policy_code = (target.policy_code + 1) % POLICY_COUNT as u8;
        target.selected_tile_id ^= u8::MAX;
        target.action = PublicActionRecord::default();
    }
    for target in &mut changed.survival_targets {
        target.initial_tile_id ^= u8::MAX;
        target.initial_wildlife = (target.initial_wildlife + 1) % 5;
        target.disposition = (target.disposition % 4) + 1;
        target.pair_survives ^= 1;
        target.final_slot = (target.final_slot + 1) % 4;
    }
    for score in &mut changed.final_scores {
        *score = score.wrapping_add(1);
    }
    if changed.model_input_bytes() != expected_input {
        return Err(
            "model input changes when provenance, physical identity, or future labels change"
                .into(),
        );
    }
    Ok(())
}

fn corpus_totals(datasets: &[AuditedDataset]) -> CorpusTotals {
    datasets
        .iter()
        .fold(CorpusTotals::default(), |mut total, dataset| {
            total.games += dataset.report.games;
            total.records += dataset.report.records;
            total.shards += dataset.report.shards;
            total.unique_model_inputs += dataset.report.unique_model_inputs;
            total.duplicate_model_inputs_within_datasets += dataset.report.duplicate_model_inputs;
            total.identity_exclusion_checks += dataset.report.identity_exclusion_checks;
            total
        })
}

fn model_input_overlaps(datasets: &[AuditedDataset]) -> Vec<ModelInputOverlap> {
    let mut overlaps = Vec::new();
    for (index, left) in datasets.iter().enumerate() {
        for right in &datasets[index + 1..] {
            let samples = left
                .model_input_hashes
                .intersection(&right.model_input_hashes)
                .take(8)
                .map(hex_digest)
                .collect::<Vec<_>>();
            let exact_hash_overlap = left
                .model_input_hashes
                .intersection(&right.model_input_hashes)
                .count();
            overlaps.push(ModelInputOverlap {
                left: left.report.role,
                right: right.report.role,
                exact_hash_overlap,
                sample_hashes: samples,
            });
        }
    }
    overlaps
}

fn combined_action_support(datasets: &[AuditedDataset], roles: &[DatasetRole]) -> ActionSupport {
    let mut combined = ActionSupport::default();
    for dataset in datasets {
        if roles.contains(&dataset.report.role) {
            combined.merge(&dataset.report.target_action_support);
        }
    }
    combined
}

fn combined_survival_support(
    datasets: &[AuditedDataset],
    roles: &[DatasetRole],
) -> SurvivalSupport {
    let mut combined = SurvivalSupport::default();
    for dataset in datasets {
        if roles.contains(&dataset.report.role) {
            combined.merge(&dataset.report.survival_support);
        }
    }
    combined
}

fn action_factor_coverage(
    training: &ActionSupport,
    evaluation: &ActionSupport,
) -> Vec<FactorCoverage> {
    let training = training.factor_classes();
    let evaluation = evaluation.factor_classes();
    training
        .iter()
        .map(|(factor, training_classes)| {
            let evaluation_classes = &evaluation[factor];
            let missing = evaluation_classes
                .difference(training_classes)
                .copied()
                .collect::<Vec<_>>();
            FactorCoverage {
                factor: (*factor).to_owned(),
                training_classes: training_classes.iter().copied().collect(),
                held_out_evaluation_classes: evaluation_classes.iter().copied().collect(),
                missing_from_training: missing.clone(),
                passed: missing.is_empty() && !evaluation_classes.is_empty(),
            }
        })
        .collect()
}

fn survival_coverage(training: &SurvivalSupport, evaluation: &SurvivalSupport) -> SurvivalCoverage {
    let training_dispositions = populated_indices(&training.disposition);
    let evaluation_dispositions = populated_indices(&evaluation.disposition);
    let training_pairs = populated_indices(&training.pair_survives);
    let evaluation_pairs = populated_indices(&evaluation.pair_survives);
    let missing_dispositions = evaluation_dispositions
        .difference(&training_dispositions)
        .copied()
        .collect::<Vec<_>>();
    let missing_pair_survival_classes = evaluation_pairs
        .difference(&training_pairs)
        .copied()
        .collect::<Vec<_>>();
    let all_dispositions = BTreeSet::from([0, 1, 2, 3]);
    let all_pairs = BTreeSet::from([0, 1]);
    let passed = missing_dispositions.is_empty()
        && missing_pair_survival_classes.is_empty()
        && training_dispositions == all_dispositions
        && evaluation_dispositions == all_dispositions
        && training_pairs == all_pairs
        && evaluation_pairs == all_pairs;
    SurvivalCoverage {
        training_dispositions: training_dispositions.into_iter().collect(),
        held_out_evaluation_dispositions: evaluation_dispositions.into_iter().collect(),
        training_pair_survival_classes: training_pairs.into_iter().collect(),
        held_out_evaluation_pair_survival_classes: evaluation_pairs.into_iter().collect(),
        missing_dispositions,
        missing_pair_survival_classes,
        passed,
    }
}

fn gates(
    datasets: &[AuditedDataset],
    totals: &CorpusTotals,
    overlaps: &[ModelInputOverlap],
    action_coverage: &[FactorCoverage],
    survival_coverage: &SurvivalCoverage,
) -> Vec<Gate> {
    let train_policy_counts = combined_policy_counts(
        datasets,
        &[DatasetRole::TrainPart0, DatasetRole::TrainPart1],
    );
    let validation = dataset(datasets, DatasetRole::Validation);
    let test = dataset(datasets, DatasetRole::Test);
    let stress = dataset(datasets, DatasetRole::FinalStress);
    let policy_boundary_passed = train_policy_counts[0] == 0
        && train_policy_counts[4] == 0
        && train_policy_counts[5] == 0
        && train_policy_counts[1..=3].iter().all(|count| *count != 0)
        && validation.report.seat_policy_counts[4] != 0
        && test.report.seat_policy_counts[5] != 0
        && stress.report.seat_policy_counts[0] != 0;
    vec![
        Gate {
            label: "Frozen corpus size and role contract",
            passed: totals.games == EXPECTED_GAMES
                && totals.records == EXPECTED_RECORDS
                && datasets.len() == DatasetRole::ALL.len(),
            observed: format!(
                "{} games, {} records, {} datasets",
                totals.games,
                totals.records,
                datasets.len()
            ),
        },
        Gate {
            label: "Exact manifest, shard, history, target, and checksum validation",
            passed: true,
            observed: format!(
                "{} immutable shards passed native dataset validation",
                totals.shards
            ),
        },
        Gate {
            label: "Policy-family holdout boundary",
            passed: policy_boundary_passed,
            observed: format!(
                "train={train_policy_counts:?}; validation competition={}; test portfolio={}; stress random={}",
                validation.report.seat_policy_counts[4],
                test.report.seat_policy_counts[5],
                stress.report.seat_policy_counts[0],
            ),
        },
        Gate {
            label: "Provenance and future-label exclusion from model input",
            passed: totals.identity_exclusion_checks == totals.records,
            observed: format!(
                "{} of {} records invariant under forbidden-field mutation",
                totals.identity_exclusion_checks, totals.records
            ),
        },
        Gate {
            label: "No exact model-input overlap between corpus roles",
            passed: overlaps
                .iter()
                .all(|overlap| overlap.exact_hash_overlap == 0),
            observed: format!(
                "{} pairwise comparisons; maximum overlap {}",
                overlaps.len(),
                overlaps
                    .iter()
                    .map(|overlap| overlap.exact_hash_overlap)
                    .max()
                    .unwrap_or(0)
            ),
        },
        Gate {
            label: "Held-out action-factor classes covered by training",
            passed: action_coverage.iter().all(|coverage| coverage.passed),
            observed: format!(
                "{} of {} factors fully covered",
                action_coverage
                    .iter()
                    .filter(|coverage| coverage.passed)
                    .count(),
                action_coverage.len()
            ),
        },
        Gate {
            label: "All tile-survival and pair-survival classes supported",
            passed: survival_coverage.passed,
            observed: format!(
                "train dispositions {:?}, held-out dispositions {:?}, train pair {:?}, held-out pair {:?}",
                survival_coverage.training_dispositions,
                survival_coverage.held_out_evaluation_dispositions,
                survival_coverage.training_pair_survival_classes,
                survival_coverage.held_out_evaluation_pair_survival_classes,
            ),
        },
    ]
}

fn scope_limitations(datasets: &[AuditedDataset]) -> Vec<ScopeLimitation> {
    let mut all_actions = ActionSupport::default();
    for dataset in datasets {
        all_actions.merge(&dataset.report.target_action_support);
    }
    let positive_paid_wipes = all_actions
        .paid_wipe_count
        .iter()
        .filter(|(count, _)| **count != 0)
        .map(|(_, observations)| *observations)
        .sum::<u64>();
    vec![
        ScopeLimitation {
            label: "Paid wildlife-wipe intent is unsupported",
            observed: format!(
                "{positive_paid_wipes} of {} target actions contain a paid wipe",
                all_actions.actions
            ),
            consequence: "The corpus may train draft and market-survival heads, but it cannot support a positive paid-wipe intent class or claims about policies that actively spend nature tokens on wildlife wipes.",
        },
        ScopeLimitation {
            label: "Strategy-switch targets are unavailable",
            observed: "All five manifests deliberately declare strategy_switch_targets_available=false."
                .to_owned(),
            consequence: "This corpus cannot supervise within-game latent strategy switching; that remains a separate successor experiment.",
        },
        ScopeLimitation {
            label: "Policy holdout covers v2 heuristic families only",
            observed: "Train, validation, test, and stress use registered v2 simulation policies rather than the v1 champion or a learned v2 policy."
                .to_owned(),
            consequence: "Passing policy-held-out calibration authorizes matched learnability research, not direct generalization claims against champion-like opponents.",
        },
    ]
}

fn dataset(datasets: &[AuditedDataset], role: DatasetRole) -> &AuditedDataset {
    datasets
        .iter()
        .find(|dataset| dataset.report.role == role)
        .expect("every frozen role was audited")
}

fn combined_policy_counts(
    datasets: &[AuditedDataset],
    roles: &[DatasetRole],
) -> [u64; POLICY_COUNT] {
    let mut counts = [0; POLICY_COUNT];
    for dataset in datasets {
        if roles.contains(&dataset.report.role) {
            merge_array(&mut counts, &dataset.report.seat_policy_counts);
        }
    }
    counts
}

fn checked_index(value: u8, length: usize, label: &str) -> Result<usize, Box<dyn Error>> {
    let index = usize::from(value);
    if index >= length {
        return Err(format!("{label} value {value} is out of range").into());
    }
    Ok(index)
}

fn merge_array<const N: usize>(left: &mut [u64; N], right: &[u64; N]) {
    for (left, right) in left.iter_mut().zip(right) {
        *left += right;
    }
}

fn merge_map(left: &mut BTreeMap<u8, u64>, right: &BTreeMap<u8, u64>) {
    for (class, count) in right {
        *left.entry(*class).or_default() += count;
    }
}

fn populated_indices<const N: usize>(values: &[u64; N]) -> BTreeSet<u8> {
    values
        .iter()
        .enumerate()
        .filter_map(|(index, value)| (*value != 0).then_some(index as u8))
        .collect()
}

fn digest_hash_set(hashes: &BTreeSet<[u8; 32]>) -> String {
    let mut hasher = Hasher::new();
    hasher.update(b"cascadia-v2-o1-model-input-set-v1");
    for hash in hashes {
        hasher.update(hash);
    }
    hasher.finalize().to_hex().to_string()
}

fn hex_digest(digest: &[u8; 32]) -> String {
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn command_output(program: &str, arguments: &[&str]) -> Option<String> {
    let output = Command::new(program).args(arguments).output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
        .filter(|value| !value.is_empty())
}

fn write_json_atomically(path: &Path, value: &impl Serialize) -> Result<(), Box<dyn Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension("json.tmp");
    let mut file = File::create(&temporary)?;
    serde_json::to_writer_pretty(&mut file, value)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    drop(file);
    fs::rename(temporary, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn action(
        draft_kind: u8,
        tile_slot: u8,
        wildlife_slot: u8,
        paid_wipe_count: u8,
    ) -> PublicActionRecord {
        PublicActionRecord {
            draft_kind,
            tile_slot,
            wildlife_slot,
            rotation: 3,
            drafted_wildlife: 2,
            wildlife_present: 1,
            replace_three_of_a_kind: 0,
            paid_wipe_count,
            paid_wipe_total_slots: paid_wipe_count,
            ..PublicActionRecord::default()
        }
    }

    #[test]
    fn parser_requires_every_frozen_role_exactly_once() {
        let args = parse_args([
            "--dataset",
            "train-part-0=/tmp/a",
            "--dataset",
            "train-part-1=/tmp/b",
            "--dataset",
            "validation=/tmp/c",
            "--dataset",
            "test=/tmp/d",
            "--dataset",
            "final-stress=/tmp/e",
            "--output",
            "/tmp/report.json",
        ])
        .unwrap();
        assert_eq!(args.datasets.len(), 5);
        assert_eq!(args.output, PathBuf::from("/tmp/report.json"));
    }

    #[test]
    fn parser_rejects_missing_or_duplicate_roles() {
        let missing = parse_args([
            "--dataset",
            "train-part-0=/tmp/a",
            "--output",
            "/tmp/report.json",
        ])
        .unwrap_err();
        assert!(missing.to_string().contains("missing dataset roles"));

        let duplicate = parse_args([
            "--dataset",
            "train-part-0=/tmp/a",
            "--dataset",
            "train-part-0=/tmp/b",
        ])
        .unwrap_err();
        assert!(duplicate.to_string().contains("duplicate"));
    }

    #[test]
    fn action_coverage_detects_unseen_held_out_class() {
        let mut training = ActionSupport::default();
        training.observe(action(0, 0, 0, 0)).unwrap();
        let mut evaluation = training.clone();
        evaluation.observe(action(1, 1, 2, 1)).unwrap();

        let coverage = action_factor_coverage(&training, &evaluation);

        assert!(
            coverage
                .iter()
                .any(|factor| factor.factor == "draft_kind" && factor.missing_from_training == [1])
        );
        assert!(coverage.iter().any(|factor| !factor.passed));
    }

    #[test]
    fn survival_coverage_requires_all_preregistered_classes() {
        let complete = SurvivalSupport {
            targets: 6,
            disposition: [1, 1, 1, 3],
            pair_survives: [4, 2],
            final_slot_when_survived: [1, 1, 1, 0],
            initial_wildlife: [1, 1, 1, 1, 2],
        };
        assert!(survival_coverage(&complete, &complete).passed);

        let incomplete = SurvivalSupport {
            disposition: [1, 0, 1, 3],
            ..complete.clone()
        };
        assert!(!survival_coverage(&incomplete, &complete).passed);
    }
}
