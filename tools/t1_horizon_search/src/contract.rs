use std::{
    collections::BTreeMap,
    error::Error,
    fs,
    io::Read,
    path::{Path, PathBuf},
};

use blake3::Hasher;
use cascadia_search::{HorizonPatternPolicyConfig, SequentialHalvingSchedule};
use clap::Parser;
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub(crate) type AnyError = Box<dyn Error + Send + Sync>;

pub(crate) const EXPERIMENT_ID: &str = "t1-search-horizon-decomposition-v1";
pub(crate) const PROTOCOL_ID: &str = "t1-strict-train-horizon-decomposition-v1";
pub(crate) const COHORT_WIDTH: usize = 64;
pub(crate) const EXPECTED_GROUPS: usize = 560;
pub(crate) const SEARCH_TRAJECTORIES_PER_GROUP: usize = 640;
pub(crate) const MODEL_BATCH_ROWS: usize = 4_096;

#[derive(Debug, Parser)]
#[command(about = "Run the preregistered T1 search-horizon decomposition")]
pub struct Args {
    #[arg(long)]
    pub(crate) dataset_root: PathBuf,
    #[arg(long)]
    pub(crate) cohort_root: PathBuf,
    #[arg(long)]
    pub(crate) model_dir: PathBuf,
    #[arg(long)]
    pub(crate) python: PathBuf,
    #[arg(long)]
    pub(crate) authorization: PathBuf,
    #[arg(long)]
    pub(crate) bundle_id: String,
    #[arg(long)]
    pub(crate) role: String,
    #[arg(long)]
    pub(crate) host: String,
    #[arg(long)]
    pub(crate) run_dir: PathBuf,
    #[arg(long)]
    pub(crate) output: PathBuf,
    #[arg(long)]
    pub(crate) maximum_groups: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum HorizonArm {
    H0RootLeaf,
    H1OneOpponent,
    H2TwoOpponents,
    H3FullRotation,
}

impl HorizonArm {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::H0RootLeaf => "h0-root-leaf",
            Self::H1OneOpponent => "h1-one-opponent",
            Self::H2TwoOpponents => "h2-two-opponents",
            Self::H3FullRotation => "h3-full-rotation",
        }
    }

    pub(crate) fn parse(value: &str) -> Result<Self, AnyError> {
        match value {
            "h0-root-leaf" => Ok(Self::H0RootLeaf),
            "h1-one-opponent" => Ok(Self::H1OneOpponent),
            "h2-two-opponents" => Ok(Self::H2TwoOpponents),
            "h3-full-rotation" => Ok(Self::H3FullRotation),
            _ => Err(format!("unknown T1 horizon arm {value}").into()),
        }
    }

    pub(crate) fn opponent_turns(self) -> usize {
        match self {
            Self::H0RootLeaf => 0,
            Self::H1OneOpponent => 1,
            Self::H2TwoOpponents => 2,
            Self::H3FullRotation => 3,
        }
    }

    pub(crate) fn expected_evaluations(self) -> usize {
        if self == Self::H0RootLeaf {
            COHORT_WIDTH
        } else {
            SEARCH_TRAJECTORIES_PER_GROUP
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct FrozenProtocol {
    root_candidates: usize,
    h0_evaluations_per_group: usize,
    stage_additional_samples: Vec<usize>,
    stage_retain: Vec<usize>,
    trajectories_per_search_group: usize,
    horizon_opponent_turns: BTreeMap<String, usize>,
    control_temperature_milli: u16,
    pattern_config: FrozenPatternConfig,
    leaf_model: String,
    leaf_value: String,
    root_chance_policy: String,
    hidden_order_policy: String,
    prefix_coupling: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct FrozenPatternConfig {
    immediate_candidate_limit: usize,
    habitat_candidate_limit: usize,
    bear_candidate_limit: usize,
    future_market_draws: usize,
}

impl FrozenProtocol {
    pub(crate) fn expected() -> Self {
        let policy = HorizonPatternPolicyConfig::default();
        let schedule = search_schedule();
        let horizon_opponent_turns = [
            HorizonArm::H0RootLeaf,
            HorizonArm::H1OneOpponent,
            HorizonArm::H2TwoOpponents,
            HorizonArm::H3FullRotation,
        ]
        .into_iter()
        .map(|arm| (arm.as_str().to_owned(), arm.opponent_turns()))
        .collect();
        Self {
            root_candidates: COHORT_WIDTH,
            h0_evaluations_per_group: COHORT_WIDTH,
            stage_additional_samples: schedule.additional_samples,
            stage_retain: schedule.retained_roots,
            trajectories_per_search_group: SEARCH_TRAJECTORIES_PER_GROUP,
            horizon_opponent_turns,
            control_temperature_milli: policy.temperature_milli,
            pattern_config: FrozenPatternConfig {
                immediate_candidate_limit: policy.blueprint.immediate_candidate_limit,
                habitat_candidate_limit: policy.blueprint.habitat_candidate_limit,
                bear_candidate_limit: policy.blueprint.bear_candidate_limit,
                future_market_draws: policy.blueprint.future_market_draws,
            },
            leaf_model: "qualified-legacy-v4opp-exact-mlx-v1".to_owned(),
            leaf_value: "v2-current-base-score-plus-legacy-nnue-remaining-value".to_owned(),
            root_chance_policy: "apply-frozen-complete-root-before-future-redeterminization"
                .to_owned(),
            hidden_order_policy:
                "arm-independent-sort-and-redeterminize-post-root-by-group-root-sample".to_owned(),
            prefix_coupling: "h1-prefix-of-h2-prefix-of-h3-by-shared-opponent-uniforms".to_owned(),
        }
    }
}

pub(crate) fn search_schedule() -> SequentialHalvingSchedule {
    SequentialHalvingSchedule {
        additional_samples: vec![4, 4, 8, 16],
        retained_roots: vec![32, 16, 8, 1],
    }
}

#[derive(Debug, Deserialize)]
pub(crate) struct Authorization {
    pub(crate) schema_version: u16,
    pub(crate) experiment_id: String,
    pub(crate) protocol_id: String,
    pub(crate) authorization_id: String,
    pub(crate) bundle_id: String,
    pub(crate) roles: BTreeMap<String, String>,
    pub(crate) protocol: FrozenProtocol,
    pub(crate) inputs: AuthorizationInputs,
}

#[derive(Debug, Deserialize)]
pub(crate) struct AuthorizationInputs {
    pub(crate) dataset_id: String,
    pub(crate) dataset_manifest_blake3: String,
    pub(crate) cohort_id: String,
    pub(crate) cohort_manifest_blake3: String,
    pub(crate) model_manifest_blake3: String,
    pub(crate) model_safetensors_blake3: String,
}

#[derive(Debug)]
pub(crate) struct CohortData {
    pub(crate) cohort_id: String,
    pub(crate) dataset_id: String,
    pub(crate) groups: usize,
    pub(crate) group_ids: Vec<u64>,
    pub(crate) game_indices: Vec<u64>,
    pub(crate) turns: Vec<u8>,
    pub(crate) current_players: Vec<u8>,
    pub(crate) source_candidate_indices: Vec<u16>,
    pub(crate) base_ranks: Vec<u16>,
    pub(crate) base_scores: Vec<f32>,
    pub(crate) action_hashes: Vec<[u8; 32]>,
    pub(crate) direct_cohort_indices: Vec<u16>,
}

pub(crate) fn load_cohort(root: &Path) -> Result<CohortData, AnyError> {
    let manifest = read_json_value(&root.join("cohort.json"))?;
    let groups = usize_value(&manifest["groups"], "T1 cohort groups")?;
    if string_value(&manifest["experiment_id"], "T1 cohort experiment")? != EXPERIMENT_ID
        || string_value(&manifest["protocol_id"], "T1 cohort protocol")?
            != "t1-strict-train-top64-cohort-v1"
        || string_value(&manifest["cohort_schema"], "T1 cohort schema")?
            != "t1-strict-exact-r2-top64-cohort-v1"
        || manifest["complete_train_corpus"].as_bool() != Some(true)
        || groups != EXPECTED_GROUPS
    {
        return Err("T1 cohort manifest violates the frozen production contract".into());
    }
    let files = manifest["files"]
        .as_object()
        .ok_or("T1 cohort files must be an object")?;
    for (name, spec) in files {
        verify_tensor_file(root, name, spec)?;
    }
    let group_ids = read_u64_tensor(root, files, "group_ids")?;
    let game_indices = read_u64_tensor(root, files, "game_indices")?;
    let turns = read_u8_tensor(root, files, "turns")?;
    let current_players = read_u8_tensor(root, files, "current_players")?;
    let source_candidate_indices = read_u16_tensor(root, files, "source_candidate_indices")?;
    let base_ranks = read_u16_tensor(root, files, "base_ranks")?;
    let base_scores = read_f32_tensor(root, files, "base_scores")?;
    let direct_cohort_indices = read_u16_tensor(root, files, "direct_cohort_indices")?;
    let action_bytes = read_tensor_bytes(root, files, "action_hashes")?;
    if group_ids.len() != groups
        || game_indices.len() != groups
        || turns.len() != groups
        || current_players.len() != groups
        || source_candidate_indices.len() != groups * COHORT_WIDTH
        || base_ranks.len() != groups * COHORT_WIDTH
        || base_scores.len() != groups * COHORT_WIDTH
        || direct_cohort_indices.len() != groups
        || action_bytes.len() != groups * COHORT_WIDTH * 32
        || base_scores.iter().any(|value| !value.is_finite())
    {
        return Err("T1 cohort tensor dimensions drifted".into());
    }
    for row in 0..groups {
        let ranks = &base_ranks[row * COHORT_WIDTH..(row + 1) * COHORT_WIDTH];
        let mut sorted = ranks.to_vec();
        sorted.sort_unstable();
        if sorted != (0..COHORT_WIDTH as u16).collect::<Vec<_>>() {
            return Err(format!("T1 cohort row {row} is not strict top 64").into());
        }
        let direct = usize::from(direct_cohort_indices[row]);
        if direct >= COHORT_WIDTH || ranks[direct] != 0 {
            return Err(format!("T1 cohort row {row} direct action drifted").into());
        }
    }
    let action_hashes = action_bytes
        .chunks_exact(32)
        .map(|bytes| bytes.try_into().expect("action hash width is exact"))
        .collect();
    Ok(CohortData {
        cohort_id: string_value(&manifest["cohort_id"], "T1 cohort ID")?.to_owned(),
        dataset_id: string_value(&manifest["dataset_id"], "T1 cohort dataset ID")?.to_owned(),
        groups,
        group_ids,
        game_indices,
        turns,
        current_players,
        source_candidate_indices,
        base_ranks,
        base_scores,
        action_hashes,
        direct_cohort_indices,
    })
}

fn verify_tensor_file(root: &Path, name: &str, spec: &Value) -> Result<(), AnyError> {
    let file = string_value(&spec["file"], "T1 tensor filename")?;
    let relative = Path::new(file);
    if relative.components().count() != 1 {
        return Err(format!("T1 tensor {name} escapes the cohort root").into());
    }
    let path = root.join(relative);
    let expected_bytes = usize_value(&spec["bytes"], "T1 tensor bytes")?;
    if !path.is_file()
        || path.metadata()?.len() != expected_bytes as u64
        || checksum(&path)? != string_value(&spec["blake3"], "T1 tensor checksum")?
    {
        return Err(format!("T1 tensor {name} failed integrity").into());
    }
    Ok(())
}

fn tensor_path(
    root: &Path,
    files: &serde_json::Map<String, Value>,
    name: &str,
) -> Result<PathBuf, AnyError> {
    let spec = files
        .get(name)
        .ok_or_else(|| format!("T1 cohort omitted tensor {name}"))?;
    Ok(root.join(string_value(&spec["file"], "T1 tensor filename")?))
}

fn read_tensor_bytes(
    root: &Path,
    files: &serde_json::Map<String, Value>,
    name: &str,
) -> Result<Vec<u8>, AnyError> {
    Ok(fs::read(tensor_path(root, files, name)?)?)
}

fn read_u8_tensor(
    root: &Path,
    files: &serde_json::Map<String, Value>,
    name: &str,
) -> Result<Vec<u8>, AnyError> {
    read_tensor_bytes(root, files, name)
}

fn read_u16_tensor(
    root: &Path,
    files: &serde_json::Map<String, Value>,
    name: &str,
) -> Result<Vec<u16>, AnyError> {
    let bytes = read_tensor_bytes(root, files, name)?;
    if bytes.len() % 2 != 0 {
        return Err(format!("T1 tensor {name} has an odd byte count").into());
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes(chunk.try_into().expect("two bytes")))
        .collect())
}

fn read_u64_tensor(
    root: &Path,
    files: &serde_json::Map<String, Value>,
    name: &str,
) -> Result<Vec<u64>, AnyError> {
    let bytes = read_tensor_bytes(root, files, name)?;
    if bytes.len() % 8 != 0 {
        return Err(format!("T1 tensor {name} byte count is not divisible by eight").into());
    }
    Ok(bytes
        .chunks_exact(8)
        .map(|chunk| u64::from_le_bytes(chunk.try_into().expect("eight bytes")))
        .collect())
}

fn read_f32_tensor(
    root: &Path,
    files: &serde_json::Map<String, Value>,
    name: &str,
) -> Result<Vec<f32>, AnyError> {
    let bytes = read_tensor_bytes(root, files, name)?;
    if bytes.len() % 4 != 0 {
        return Err(format!("T1 tensor {name} byte count is not divisible by four").into());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four bytes")))
        .collect())
}

pub(crate) fn validate_authorization(
    authorization: &Authorization,
    args: &Args,
) -> Result<(), AnyError> {
    if authorization.schema_version != 1
        || authorization.experiment_id != EXPERIMENT_ID
        || authorization.protocol_id != PROTOCOL_ID
        || authorization.bundle_id != args.bundle_id
        || authorization.protocol != FrozenProtocol::expected()
        || !is_digest(&authorization.authorization_id)
        || !authorization.roles.contains_key(&args.role)
    {
        return Err("T1 authorization does not match the frozen protocol".into());
    }
    Ok(())
}

pub(crate) fn checksum(path: &Path) -> Result<String, AnyError> {
    let mut file = fs::File::open(path)?;
    let mut hasher = Hasher::new();
    let mut buffer = [0u8; 1 << 20];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hasher.finalize().to_hex().to_string())
}

pub(crate) fn canonical_blake3(value: &Value) -> Result<String, AnyError> {
    Ok(blake3::hash(&serde_json::to_vec(value)?)
        .to_hex()
        .to_string())
}

pub(crate) fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, AnyError> {
    Ok(serde_json::from_reader(fs::File::open(path)?)?)
}

pub(crate) fn read_json_value(path: &Path) -> Result<Value, AnyError> {
    read_json(path)
}

pub(crate) fn write_json_atomic(path: &Path, value: &impl Serialize) -> Result<(), AnyError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension(format!(
        "{}.tmp",
        path.extension()
            .and_then(|value| value.to_str())
            .unwrap_or("")
    ));
    fs::write(&temporary, serde_json::to_vec_pretty(value)?)?;
    fs::rename(temporary, path)?;
    Ok(())
}

pub(crate) fn string_value<'a>(value: &'a Value, field: &str) -> Result<&'a str, AnyError> {
    value
        .as_str()
        .ok_or_else(|| format!("{field} is not a string").into())
}

pub(crate) fn usize_value(value: &Value, field: &str) -> Result<usize, AnyError> {
    usize::try_from(
        value
            .as_u64()
            .ok_or_else(|| format!("{field} is not an unsigned integer"))?,
    )
    .map_err(Into::into)
}

fn is_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|character| character.is_ascii_hexdigit() && !character.is_ascii_uppercase())
}
