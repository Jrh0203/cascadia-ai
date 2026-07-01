//! Immutable-container CPU worker for Cascadia V3 collection and validation.
//!
//! Bacalhau assigns disjoint game-index ranges. The worker never selects a
//! host and writes one compact, replay-authoritative shard plus a checksummed
//! receipt into its declared output directory.

use std::{
    collections::BTreeMap,
    fs,
    io::Read,
    path::{Path, PathBuf},
    time::Instant,
};

use cascadia_differential::legacy_teacher::{
    LEGACY_DIRECT_POLICY_STRATEGY_ID, LegacyDirectPolicy, LegacyTeacher, load_legacy_weights,
};
use cascadia_game::{GameConfig, GameSeed, GameState, Replay, TurnAction, score_game};
use cascadia_sim::{select_greedy_action, strategy_rng};
use cascadia_v3_nnue::{
    InferenceBackend, QuantizedV3Model, TerminalRolloutConfig, V3CampaignState, V3GameRecord,
    V3GameShardReader, V3GameShardWriter, V3LabeledTeacherRoot, V3LabeledTeacherRootShardWriter,
    V3SearchBudget, V3SearchPolicy, V3TeacherCandidateEstimate, V3TeacherRootLabel,
    V3TeacherRootShardReader, V3TrainingProvenance, encode_public_features, select_boltzmann_top32,
};
use clap::{Parser, Subcommand, ValueEnum};
use rand::{distributions::WeightedIndex, prelude::Distribution};
use rand_chacha::ChaCha8Rng;
use serde::Serialize;

#[derive(Debug, Parser)]
#[command(about = "Canonical Cascadia V3 Bacalhau CPU worker")]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Collect {
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        games: usize,
        #[arg(long)]
        first_game_index: u64,
        #[arg(long, value_enum)]
        component: CollectionComponent,
        #[arg(long)]
        v1_weights: Option<PathBuf>,
        #[arg(long)]
        v3_model_dir: Vec<PathBuf>,
        #[arg(long)]
        cycle: Option<u8>,
        #[arg(long, default_value_t = 0.1)]
        epsilon: f64,
        #[arg(long, default_value_t = 1.0)]
        temperature: f64,
        #[arg(long)]
        campaign_state: Option<PathBuf>,
        #[arg(long)]
        approved_readiness_sha256: Option<String>,
    },
    VerifyShard {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    LabelRoots {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        v1_weights: PathBuf,
        #[arg(long, default_value_t = 600)]
        rollouts: usize,
        #[arg(long)]
        cycle: Option<u8>,
        #[arg(long)]
        campaign_state: PathBuf,
        #[arg(long)]
        approved_readiness_sha256: String,
    },
    PromotionPairs {
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        treatment_model_dir: PathBuf,
        #[arg(long)]
        control_model_dir: PathBuf,
        #[arg(long)]
        v1_weights: PathBuf,
        #[arg(long, value_enum)]
        tier: PromotionTier,
        #[arg(long)]
        first_pair_index: usize,
        #[arg(long)]
        pairs: usize,
        #[arg(long)]
        cycle: u8,
        #[arg(long)]
        campaign_state: PathBuf,
        #[arg(long)]
        approved_readiness_sha256: String,
    },
    FinalProtectedPairs {
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        treatment_model_dir: PathBuf,
        #[arg(long)]
        v1_weights: PathBuf,
        #[arg(long)]
        first_pair_index: usize,
        #[arg(long)]
        pairs: usize,
        #[arg(long)]
        seed_domain_key: String,
        #[arg(long)]
        campaign_state: PathBuf,
        #[arg(long)]
        approved_readiness_sha256: String,
    },
    FinalAllV3 {
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        model_dir: PathBuf,
        #[arg(long)]
        first_game_index: usize,
        #[arg(long)]
        games: usize,
        #[arg(long)]
        seed_domain_key: String,
        #[arg(long)]
        campaign_state: PathBuf,
        #[arg(long)]
        approved_readiness_sha256: String,
    },
    Health {
        #[arg(long)]
        output: PathBuf,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Serialize)]
#[serde(rename_all = "kebab-case")]
enum CollectionComponent {
    EngineeringSmoke,
    EngineeringExpertSmoke,
    Greedy,
    V1Direct,
    MixedFrozen,
    RareSoftmax,
    ExpertIteration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Serialize)]
#[serde(rename_all = "kebab-case")]
enum PromotionTier {
    Direct,
    K32R64,
    K32R600,
    EqualWallTime,
}

#[derive(Debug, Clone, Copy)]
enum SeatPolicy {
    Greedy,
    V1,
    V3(usize),
    Rare,
}

fn checksum(path: &Path) -> Result<String, std::io::Error> {
    let mut input = fs::File::open(path)?;
    let mut hasher = blake3::Hasher::new();
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let count = input.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(hasher.finalize().to_hex().to_string())
}

fn write_json_atomic(
    path: &Path,
    value: &impl Serialize,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension(format!("tmp-{}", std::process::id()));
    fs::write(&temporary, serde_json::to_vec_pretty(value)?)?;
    fs::rename(temporary, path)?;
    Ok(())
}

fn load_v3_models(
    directories: &[PathBuf],
) -> Result<(Vec<QuantizedV3Model>, Vec<String>), Box<dyn std::error::Error>> {
    let mut models = Vec::with_capacity(directories.len());
    let mut identities = Vec::with_capacity(directories.len());
    for directory in directories {
        let (model, manifest) = QuantizedV3Model::load_bundle(directory)?;
        identities.push(format!(
            "{}:{}:{}",
            manifest.architecture_id, manifest.checkpoint_id, manifest.weights_blake3
        ));
        models.push(model);
    }
    Ok((models, identities))
}

fn rare_action(
    game: &GameState,
    rng: &mut ChaCha8Rng,
    temperature: f64,
) -> Result<TurnAction, Box<dyn std::error::Error>> {
    if !temperature.is_finite() || temperature <= 0.0 {
        return Err("rare-action temperature must be finite and positive".into());
    }
    let (prelude, _) = game.preview_free_three_of_a_kind_if_feasible()?;
    let actions = game.legal_turn_actions(&prelude)?;
    if actions.is_empty() {
        return Err("rare-action policy found no legal action".into());
    }
    let weights = actions
        .iter()
        .map(|action| {
            let independent =
                matches!(action.draft, cascadia_game::DraftChoice::Independent { .. });
            let novelty = f64::from(action.tile.coord.radius()) * 0.18
                + if independent { 2.5 } else { 0.0 }
                + action.wildlife_wipes.len() as f64 * 2.0
                + if action.replace_three_of_a_kind {
                    0.75
                } else {
                    0.0
                }
                + if action.wildlife.is_none() { 0.35 } else { 0.0 };
            (novelty / temperature).exp()
        })
        .collect::<Vec<_>>();
    let distribution = WeightedIndex::new(&weights)?;
    Ok(actions[distribution.sample(rng)].clone())
}

fn choose_seat_policies(
    component: CollectionComponent,
    game_index: u64,
    has_v1: bool,
    v3_models: usize,
) -> Result<[SeatPolicy; 4], Box<dyn std::error::Error>> {
    match component {
        CollectionComponent::EngineeringSmoke => Ok([SeatPolicy::Greedy; 4]),
        CollectionComponent::EngineeringExpertSmoke => {
            if v3_models == 0 || !has_v1 {
                return Err("engineering expert smoke requires V3 and V1 models".into());
            }
            let focal = game_index as usize % 4;
            Ok(std::array::from_fn(|seat| {
                if seat == focal {
                    SeatPolicy::V3(0)
                } else {
                    SeatPolicy::V1
                }
            }))
        }
        CollectionComponent::Greedy => Ok([SeatPolicy::Greedy; 4]),
        CollectionComponent::V1Direct => {
            if !has_v1 {
                return Err("v1-direct collection requires --v1-weights".into());
            }
            Ok([SeatPolicy::V1; 4])
        }
        CollectionComponent::RareSoftmax => Ok([SeatPolicy::Rare; 4]),
        CollectionComponent::MixedFrozen => {
            let mut pool = vec![SeatPolicy::Greedy];
            if has_v1 {
                pool.push(SeatPolicy::V1);
            }
            pool.extend((0..v3_models).map(SeatPolicy::V3));
            if pool.len() < 2 {
                return Err("mixed-frozen collection requires at least two frozen policies".into());
            }
            Ok(std::array::from_fn(|seat| {
                pool[(game_index as usize * 5 + seat * 3) % pool.len()]
            }))
        }
        CollectionComponent::ExpertIteration => {
            if v3_models == 0 || !has_v1 {
                return Err(
                    "expert iteration requires newest V3 model first and frozen V1 weights".into(),
                );
            }
            let focal = game_index as usize % 4;
            Ok(std::array::from_fn(|seat| {
                if seat == focal {
                    SeatPolicy::V3(0)
                } else if v3_models == 1 || (game_index as usize * 7 + seat * 3) % 10 < 8 {
                    SeatPolicy::V1
                } else {
                    SeatPolicy::V3(1 + (game_index as usize + seat * 7) % (v3_models - 1))
                }
            }))
        }
    }
}

fn policy_id(policy: SeatPolicy, v3_ids: &[String]) -> String {
    match policy {
        SeatPolicy::Greedy => "cascadia-v3-greedy-v1".to_owned(),
        SeatPolicy::V1 => LEGACY_DIRECT_POLICY_STRATEGY_ID.to_owned(),
        SeatPolicy::V3(index) => v3_ids[index].clone(),
        SeatPolicy::Rare => "cascadia-v3-rare-legal-softmax-v1".to_owned(),
    }
}

fn authorize_collection(
    component: CollectionComponent,
    cycle: Option<u8>,
    campaign_state: Option<&Path>,
    approved_readiness_sha256: Option<&str>,
) -> Result<Option<V3CampaignState>, Box<dyn std::error::Error>> {
    if matches!(
        component,
        CollectionComponent::EngineeringSmoke | CollectionComponent::EngineeringExpertSmoke
    ) {
        if campaign_state.is_some() || approved_readiness_sha256.is_some() {
            return Err("engineering smoke must not claim Phase 2 authorization".into());
        }
        return Ok(None);
    }
    let path = campaign_state.ok_or("scientific collection requires --campaign-state")?;
    let approved = approved_readiness_sha256
        .ok_or("scientific collection requires --approved-readiness-sha256")?;
    let state = V3CampaignState::load_verified(path)?;
    let expected_phase = match component {
        CollectionComponent::ExpertIteration => {
            format!(
                "cycle-{:02}-collecting",
                cycle.ok_or("expert cycle is missing")?
            )
        }
        CollectionComponent::Greedy
        | CollectionComponent::V1Direct
        | CollectionComponent::MixedFrozen
        | CollectionComponent::RareSoftmax => "bootstrap_collecting".to_owned(),
        CollectionComponent::EngineeringSmoke | CollectionComponent::EngineeringExpertSmoke => {
            unreachable!()
        }
    };
    if state.part != 2
        || !state.phase2_authorized
        || state.phase != expected_phase
        || state.protected_seed_values_opened
        || state.approved_readiness_sha256.as_deref() != Some(approved)
        || state.readiness_sha256.as_deref() != Some(approved)
    {
        return Err("scientific collection is not authorized for this exact V3 phase".into());
    }
    Ok(Some(state))
}

#[allow(clippy::too_many_arguments)]
fn collect(
    output: &Path,
    games: usize,
    first_game_index: u64,
    component: CollectionComponent,
    v1_weights: Option<&Path>,
    v3_directories: &[PathBuf],
    cycle: Option<u8>,
    epsilon: f64,
    temperature: f64,
    campaign_state: Option<&Path>,
    approved_readiness_sha256: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    if games == 0 || !epsilon.is_finite() || !(0.0..=1.0).contains(&epsilon) {
        return Err("games and exploration parameters are invalid".into());
    }
    if component == CollectionComponent::ExpertIteration && !matches!(cycle, Some(1..=10)) {
        return Err("expert iteration requires --cycle in 1..=10".into());
    }
    if component != CollectionComponent::ExpertIteration && cycle.is_some() {
        return Err("--cycle is only valid for expert-iteration collection".into());
    }
    let authorization =
        authorize_collection(component, cycle, campaign_state, approved_readiness_sha256)?;
    let (v3_models, v3_ids) = load_v3_models(v3_directories)?;
    let mut v1_policy = v1_weights
        .map(load_legacy_weights)
        .transpose()?
        .map(LegacyDirectPolicy::new)
        .transpose()?;
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    let started = Instant::now();
    let mut writer = V3GameShardWriter::create(output)?;
    let mut policy_games = BTreeMap::<String, u64>::new();
    let mut focal_score_histogram = BTreeMap::<u16, u64>::new();
    let mut focal_score_sum = 0u64;
    let mut focal_wildlife_sums = [0u64; 5];
    let mut focal_terrain_sums = [0u64; 5];
    let mut focal_nature_tokens = 0u64;
    let expert_shaped = matches!(
        component,
        CollectionComponent::ExpertIteration | CollectionComponent::EngineeringExpertSmoke
    );
    for offset in 0..games {
        let game_index = first_game_index + offset as u64;
        let seed = GameSeed::from_u64(game_index);
        let policies =
            choose_seat_policies(component, game_index, v1_policy.is_some(), v3_models.len())?;
        let seat_policy_ids = std::array::from_fn(|seat| policy_id(policies[seat], &v3_ids));
        for identity in &seat_policy_ids {
            *policy_games.entry(identity.clone()).or_default() += 1;
        }
        let mut game = GameState::new(GameConfig::research_aaaaa(4)?, seed)?;
        let mut replay = Replay::new(GameConfig::research_aaaaa(4)?, seed);
        let mut rngs = (0..4)
            .map(|seat| strategy_rng(seed, seat, "cascadia-v3-campaign-collection-v1"))
            .collect::<Vec<_>>();
        while !game.is_game_over() {
            let seat = game.current_player();
            let action = match policies[seat] {
                SeatPolicy::Greedy => {
                    let (prelude, _) = game.preview_free_three_of_a_kind_if_feasible()?;
                    select_greedy_action(&game, &prelude, &mut rngs[seat])?
                }
                SeatPolicy::V1 => v1_policy
                    .as_mut()
                    .ok_or("V1 policy was not loaded")?
                    .select_action(&game)?,
                SeatPolicy::V3(index) => {
                    let policy = V3SearchPolicy::new(&v3_models[index], InferenceBackend::Neon)?;
                    let ranked = policy.rank_legal_actions(&game)?;
                    if matches!(
                        component,
                        CollectionComponent::ExpertIteration
                            | CollectionComponent::EngineeringExpertSmoke
                    ) && index == 0
                    {
                        select_boltzmann_top32(&ranked, epsilon, temperature, &mut rngs[seat])?
                    } else {
                        ranked
                            .first()
                            .ok_or("V3 policy found no action")?
                            .action
                            .clone()
                    }
                }
                SeatPolicy::Rare => rare_action(&game, &mut rngs[seat], temperature)?,
            };
            game.apply(&action)?;
            replay.turns.push(action);
        }
        replay.seal()?;
        let focal_training_seat = expert_shaped.then_some((game_index % 4) as u8);
        let newest_model_id = focal_training_seat.map(|_| v3_ids[0].clone());
        if let Some(focal) = focal_training_seat {
            let score = &score_game(&game)[usize::from(focal)];
            *focal_score_histogram.entry(score.base_total).or_default() += 1;
            focal_score_sum += u64::from(score.base_total);
            for (total, value) in focal_wildlife_sums.iter_mut().zip(score.wildlife) {
                *total += u64::from(value);
            }
            for (total, value) in focal_terrain_sums.iter_mut().zip(score.habitat) {
                *total += u64::from(value);
            }
            focal_nature_tokens += u64::from(score.nature_tokens);
        }
        writer.append(&V3GameRecord {
            game_index,
            replay,
            seat_policy_ids,
            newest_model_id,
            focal_training_seat,
            exploration_epsilon: if expert_shaped {
                epsilon as f32
            } else if component == CollectionComponent::RareSoftmax {
                1.0
            } else {
                0.0
            },
            provenance: match component {
                CollectionComponent::ExpertIteration => V3TrainingProvenance::ExpertIteration {
                    cycle: cycle.unwrap(),
                },
                CollectionComponent::EngineeringSmoke
                | CollectionComponent::EngineeringExpertSmoke => {
                    V3TrainingProvenance::EngineeringSmoke
                }
                _ => V3TrainingProvenance::Bootstrap {
                    component: format!("{component:?}").to_lowercase(),
                },
            },
        })?;
    }
    let records = writer.finish()?;
    let receipt = serde_json::json!({
        "schema_id": "cascadia-v3-collection-shard-receipt-v1",
        "scientific_eligible": !matches!(component, CollectionComponent::EngineeringSmoke | CollectionComponent::EngineeringExpertSmoke),
        "component": component,
        "cycle": cycle,
        "first_game_index": first_game_index,
        "games": games,
        "records": records,
        "seed_domain": "scheduler-assigned-game-index-v1",
        "newest_model_seats_per_expert_game": if matches!(component, CollectionComponent::ExpertIteration | CollectionComponent::EngineeringExpertSmoke) { 1 } else { 0 },
        "policy_seat_games": policy_games,
        "focal_benchmark": expert_shaped.then(|| serde_json::json!({
            "count": games,
            "score_sum": focal_score_sum,
            "score_histogram": focal_score_histogram,
            "wildlife_sums": {
                "bear": focal_wildlife_sums[0],
                "elk": focal_wildlife_sums[1],
                "salmon": focal_wildlife_sums[2],
                "hawk": focal_wildlife_sums[3],
                "fox": focal_wildlife_sums[4],
            },
            "terrain_sums": {
                "forest": focal_terrain_sums[0],
                "mountain": focal_terrain_sums[1],
                "prairie": focal_terrain_sums[2],
                "wetland": focal_terrain_sums[3],
                "river": focal_terrain_sums[4],
            },
            "nature_tokens_sum": focal_nature_tokens,
            "pinecones_sum": focal_nature_tokens,
        })),
        "shard": output,
        "bytes": output.metadata()?.len(),
        "blake3": checksum(output)?,
        "elapsed_seconds": started.elapsed().as_secs_f64(),
        "approved_readiness_sha256": authorization
            .as_ref()
            .and_then(|state| state.approved_readiness_sha256.clone()),
        "campaign_state_sha256": authorization.and_then(|state| state.state_sha256),
    });
    write_json_atomic(&output.with_extension("receipt.json"), &receipt)?;
    println!("{}", serde_json::to_string(&receipt)?);
    Ok(())
}

fn verify_shard(input: &Path, output: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let mut reader = V3GameShardReader::open(input)?;
    let declared = reader.len();
    let mut records = 0u64;
    let mut entries = 0u64;
    while let Some(record) = reader.next_record()? {
        entries += record.training_entries()?.len() as u64;
        records += 1;
    }
    if records != declared {
        return Err("compact shard record count changed during verification".into());
    }
    write_json_atomic(
        output,
        &serde_json::json!({
            "schema_id": "cascadia-v3-compact-shard-verification-v1",
            "passed": true,
            "records": records,
            "expanded_training_entries": entries,
            "bytes": input.metadata()?.len(),
            "blake3": checksum(input)?,
        }),
    )
}

fn authorize_labeling(
    cycle: Option<u8>,
    campaign_state: &Path,
    approved_readiness_sha256: &str,
) -> Result<V3CampaignState, Box<dyn std::error::Error>> {
    if cycle.is_some_and(|value| !(1..=10).contains(&value)) {
        return Err("teacher labeling cycle is outside 1..=10".into());
    }
    let state = V3CampaignState::load_verified(campaign_state)?;
    let expected_phase = cycle.map_or_else(
        || "bootstrap_labeling".to_owned(),
        |value| format!("cycle-{value:02}-labeling"),
    );
    if state.part != 2
        || !state.phase2_authorized
        || state.phase != expected_phase
        || state.protected_seed_values_opened
        || state.approved_readiness_sha256.as_deref() != Some(approved_readiness_sha256)
        || state.readiness_sha256.as_deref() != Some(approved_readiness_sha256)
    {
        return Err("teacher labeling is not authorized for this exact V3 phase".into());
    }
    Ok(state)
}

#[allow(clippy::too_many_arguments)]
fn label_roots(
    input: &Path,
    output: &Path,
    v1_weights: &Path,
    rollouts: usize,
    cycle: Option<u8>,
    campaign_state: &Path,
    approved_readiness_sha256: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    if rollouts != 600 {
        return Err("registered V3 teacher labeling requires exactly R600".into());
    }
    let authorization = authorize_labeling(cycle, campaign_state, approved_readiness_sha256)?;
    let net = load_legacy_weights(v1_weights)?;
    let mut teacher = LegacyTeacher::new_heuristic(net, rollouts)?;
    let mut reader = V3TeacherRootShardReader::open(input)?;
    let expected = reader.len();
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut writer = V3LabeledTeacherRootShardWriter::create(output)?;
    let started = Instant::now();
    let teacher_id = "qualified-v1-direct-top32-terminal-r600-sequential-halving-v1".to_owned();
    let mut roots = 0u64;
    let mut candidate_estimates = 0u64;
    while let Some(root) = reader.next_root()? {
        let game = root.reconstruct()?;
        let decision = teacher.select_action_with_exact_budget_estimates(&game)?;
        let actual_budget = decision
            .estimates
            .iter()
            .map(|estimate| estimate.samples as usize)
            .sum::<usize>();
        if actual_budget != rollouts {
            return Err(format!(
                "qualified teacher spent {actual_budget} rollouts, expected {rollouts}"
            )
            .into());
        }
        let candidates = decision
            .estimates
            .iter()
            .enumerate()
            .map(|(index, estimate)| V3TeacherCandidateEstimate {
                action: estimate.action.clone(),
                direct_raw_units: estimate.direct_raw_units,
                rollout_mean: estimate.rollout_mean,
                rollout_variance: estimate.rollout_stddev * estimate.rollout_stddev,
                rollout_count: estimate.samples,
                rank: (index + 1) as u8,
            })
            .collect::<Vec<_>>();
        candidate_estimates += candidates.len() as u64;
        let label = V3TeacherRootLabel {
            state_blake3: root.state_blake3,
            focal_seat: root.stratum.focal_seat,
            phase_bucket: root.stratum.phase_bucket,
            candidate_limit: 32,
            rollout_budget: rollouts as u16,
            selected_action: decision.selected,
            candidates,
            rng_domain: "qualified-v1-public-state-r600-v1".to_owned(),
        };
        writer.append(&V3LabeledTeacherRoot {
            root,
            teacher_id: teacher_id.clone(),
            label,
        })?;
        roots += 1;
    }
    let written = writer.finish()?;
    if roots != expected || written != expected {
        return Err("labeled teacher-root count differs from input".into());
    }
    let receipt = serde_json::json!({
        "schema_id": "cascadia-v3-teacher-label-shard-receipt-v1",
        "passed": true,
        "scientific_eligible": true,
        "teacher_id": teacher_id,
        "cycle": cycle,
        "roots": roots,
        "candidate_estimates": candidate_estimates,
        "rollouts_per_root": rollouts,
        "input": input,
        "input_bytes": input.metadata()?.len(),
        "input_blake3": checksum(input)?,
        "output": output,
        "output_bytes": output.metadata()?.len(),
        "output_blake3": checksum(output)?,
        "elapsed_seconds": started.elapsed().as_secs_f64(),
        "approved_readiness_sha256": approved_readiness_sha256,
        "campaign_state_sha256": authorization.state_sha256,
        "bridge_diagnostics": teacher.diagnostics,
    });
    write_json_atomic(&output.with_extension("receipt.json"), &receipt)?;
    println!("{}", serde_json::to_string(&receipt)?);
    Ok(())
}

fn promotion_budget(tier: PromotionTier) -> V3SearchBudget {
    match tier {
        PromotionTier::Direct => V3SearchBudget::Direct,
        PromotionTier::K32R64 => V3SearchBudget::K32R64,
        PromotionTier::K32R600 | PromotionTier::EqualWallTime => V3SearchBudget::K32R600,
    }
}

fn play_promotion_game(
    policy: &V3SearchPolicy<'_>,
    v1_policy: &mut LegacyDirectPolicy,
    seed: GameSeed,
    focal: usize,
    tier: PromotionTier,
) -> Result<(u16, f64, String), Box<dyn std::error::Error>> {
    let mut game = GameState::new(GameConfig::research_aaaaa(4)?, seed)?;
    let mut focal_seconds = 0.0;
    while !game.is_game_over() {
        let seat = game.current_player();
        let action = if seat == focal {
            let started = Instant::now();
            let action = policy.select_action(
                &game,
                promotion_budget(tier),
                TerminalRolloutConfig {
                    model_guided: false,
                    maximum_plies: None,
                },
            )?;
            focal_seconds += started.elapsed().as_secs_f64();
            action
        } else {
            v1_policy.select_action(&game)?
        };
        game.apply(&action)?;
    }
    Ok((
        score_game(&game)[focal].base_total,
        focal_seconds,
        game.canonical_hash().to_hex().to_string(),
    ))
}

#[allow(clippy::too_many_arguments)]
fn promotion_pairs(
    output: &Path,
    treatment_model_dir: &Path,
    control_model_dir: &Path,
    v1_weights: &Path,
    tier: PromotionTier,
    first_pair_index: usize,
    pairs: usize,
    cycle: u8,
    campaign_state: &Path,
    approved_readiness_sha256: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    if pairs == 0 || first_pair_index.checked_add(pairs).is_none() || !(1..=10).contains(&cycle) {
        return Err("promotion pair range or cycle is invalid".into());
    }
    let state = V3CampaignState::load_verified(campaign_state)?;
    if state.part != 2
        || !state.phase2_authorized
        || state.phase != format!("cycle-{cycle:02}-promotion")
        || state.protected_seed_values_opened
        || state.approved_readiness_sha256.as_deref() != Some(approved_readiness_sha256)
        || state.readiness_sha256.as_deref() != Some(approved_readiness_sha256)
    {
        return Err("promotion pairs are not authorized for this exact cycle".into());
    }
    let (treatment_model, treatment_manifest) = QuantizedV3Model::load_bundle(treatment_model_dir)?;
    let (control_model, control_manifest) = QuantizedV3Model::load_bundle(control_model_dir)?;
    if treatment_manifest.architecture_id != control_manifest.architecture_id {
        return Err("promotion treatment and control architectures differ".into());
    }
    let treatment_id = format!(
        "{}:{}:{}",
        treatment_manifest.architecture_id,
        treatment_manifest.checkpoint_id,
        treatment_manifest.weights_blake3
    );
    let control_id = format!(
        "{}:{}:{}",
        control_manifest.architecture_id,
        control_manifest.checkpoint_id,
        control_manifest.weights_blake3
    );
    if treatment_id == control_id {
        return Err("promotion treatment and control must be distinct checkpoints".into());
    }
    let treatment = V3SearchPolicy::new(&treatment_model, InferenceBackend::Neon)?;
    let control = V3SearchPolicy::new(&control_model, InferenceBackend::Neon)?;
    let net = load_legacy_weights(v1_weights)?;
    let mut v1_policy = LegacyDirectPolicy::new(net)?;
    let tier_index = match tier {
        PromotionTier::Direct => 0u64,
        PromotionTier::K32R64 => 1,
        PromotionTier::K32R600 => 2,
        PromotionTier::EqualWallTime => 3,
    };
    let started = Instant::now();
    let mut records = Vec::with_capacity(pairs);
    for pair_index in first_pair_index..first_pair_index + pairs {
        let raw_seed = 3_000_000_000u64
            + u64::from(cycle) * 10_000_000
            + tier_index * 1_000_000
            + pair_index as u64;
        let seed = GameSeed::from_u64(raw_seed);
        let focal = pair_index % 4;
        let (treatment_score, treatment_seconds, treatment_hash) =
            play_promotion_game(&treatment, &mut v1_policy, seed, focal, tier)?;
        let (control_score, control_seconds, control_hash) =
            play_promotion_game(&control, &mut v1_policy, seed, focal, tier)?;
        records.push(serde_json::json!({
            "tier": tier,
            "pair_index": pair_index,
            "raw_seed": raw_seed,
            "focal_seat": focal,
            "treatment_score": treatment_score,
            "control_score": control_score,
            "paired_delta": i32::from(treatment_score) - i32::from(control_score),
            "treatment_focal_seconds": treatment_seconds,
            "control_focal_seconds": control_seconds,
            "treatment_final_state_blake3": treatment_hash,
            "control_final_state_blake3": control_hash,
            "rng_domain": "cascadia-v3-promotion-paired-v1",
            "opponent_policy": LEGACY_DIRECT_POLICY_STRATEGY_ID,
            "integrity_passed": true,
        }));
    }
    let value = serde_json::json!({
        "schema_id": "cascadia-v3-promotion-pair-shard-v1",
        "passed": true,
        "scientific_eligible": true,
        "cycle": cycle,
        "tier": tier,
        "first_pair_index": first_pair_index,
        "pairs": pairs,
        "treatment_model_id": treatment_id,
        "control_model_id": control_id,
        "search_budget": promotion_budget(tier),
        "equal_wall_time_contract": if tier == PromotionTier::EqualWallTime {
            "same-architecture-same-k32-r600-budget-with-measured-focal-time"
        } else {
            "not-applicable"
        },
        "records": records,
        "elapsed_seconds": started.elapsed().as_secs_f64(),
        "approved_readiness_sha256": approved_readiness_sha256,
        "campaign_state_sha256": state.state_sha256,
    });
    write_json_atomic(output, &value)?;
    Ok(())
}

fn protected_seed(
    key: &str,
    domain: &[u8],
    index: usize,
) -> Result<(u64, GameSeed), Box<dyn std::error::Error>> {
    if key.len() != 64 || !key.bytes().all(|value| value.is_ascii_hexdigit()) {
        return Err("protected seed-domain key must be 32 hex-encoded bytes".into());
    }
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"cascadia-v3-protected-seed-v1");
    hasher.update(key.as_bytes());
    hasher.update(domain);
    hasher.update(&(index as u64).to_le_bytes());
    let digest = hasher.finalize();
    let raw = u64::from_le_bytes(digest.as_bytes()[..8].try_into().unwrap());
    Ok((raw, GameSeed::from_u64(raw)))
}

fn anatomy(score: &cascadia_game::ScoreBreakdown, overflow_states: u64) -> serde_json::Value {
    serde_json::json!({
        "bear": score.wildlife[0],
        "elk": score.wildlife[1],
        "salmon": score.wildlife[2],
        "hawk": score.wildlife[3],
        "fox": score.wildlife[4],
        "wildlife_total": score.wildlife.iter().map(|value| u64::from(*value)).sum::<u64>(),
        "forest": score.habitat[0],
        "mountain": score.habitat[1],
        "prairie": score.habitat[2],
        "wetland": score.habitat[3],
        "river": score.habitat[4],
        "terrain_total": score.habitat.iter().map(|value| u64::from(*value)).sum::<u64>(),
        "nature_tokens": score.nature_tokens,
        "pinecones": score.nature_tokens,
        "overflow_states": overflow_states,
    })
}

fn play_final_treatment(
    policy: &V3SearchPolicy<'_>,
    opponents: &mut LegacyDirectPolicy,
    seed: GameSeed,
    focal: usize,
) -> Result<(cascadia_game::ScoreBreakdown, f64, u64), Box<dyn std::error::Error>> {
    let mut game = GameState::new(GameConfig::research_aaaaa(4)?, seed)?;
    let mut seconds = 0.0;
    let mut overflow = 0u64;
    while !game.is_game_over() {
        let seat = game.current_player();
        overflow +=
            u64::from(!encode_public_features(&game.public_state(), seat)?.natural_hot_path());
        let action = if seat == focal {
            let started = Instant::now();
            let action = policy.select_action(
                &game,
                V3SearchBudget::K32R600,
                TerminalRolloutConfig {
                    model_guided: false,
                    maximum_plies: None,
                },
            )?;
            seconds += started.elapsed().as_secs_f64();
            action
        } else {
            opponents.select_action(&game)?
        };
        game.apply(&action)?;
    }
    Ok((score_game(&game)[focal], seconds, overflow))
}

fn play_final_control(
    control: &mut LegacyTeacher,
    opponents: &mut LegacyDirectPolicy,
    seed: GameSeed,
    focal: usize,
) -> Result<(cascadia_game::ScoreBreakdown, f64, u64), Box<dyn std::error::Error>> {
    let mut game = GameState::new(GameConfig::research_aaaaa(4)?, seed)?;
    let mut seconds = 0.0;
    let mut overflow = 0u64;
    while !game.is_game_over() {
        let seat = game.current_player();
        overflow +=
            u64::from(!encode_public_features(&game.public_state(), seat)?.natural_hot_path());
        let action = if seat == focal {
            let started = Instant::now();
            let action = control
                .select_action_with_exact_budget_estimates(&game)?
                .selected;
            seconds += started.elapsed().as_secs_f64();
            action
        } else {
            opponents.select_action(&game)?
        };
        game.apply(&action)?;
    }
    Ok((score_game(&game)[focal], seconds, overflow))
}

#[allow(clippy::too_many_arguments)]
fn final_protected_pairs(
    output: &Path,
    treatment_model_dir: &Path,
    v1_weights: &Path,
    first_pair_index: usize,
    pairs: usize,
    seed_domain_key: &str,
    campaign_state: &Path,
    approved_readiness_sha256: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    if pairs == 0 || first_pair_index.checked_add(pairs).is_none() {
        return Err("protected pair range is invalid".into());
    }
    let state = V3CampaignState::load_verified(campaign_state)?;
    if state.part != 2
        || state.phase != "final_protected_comparison"
        || !state.phase2_authorized
        || !state.protected_seed_values_opened
        || state.approved_readiness_sha256.as_deref() != Some(approved_readiness_sha256)
        || state.readiness_sha256.as_deref() != Some(approved_readiness_sha256)
    {
        return Err("protected comparison is not open for this campaign state".into());
    }
    let (model, manifest) = QuantizedV3Model::load_bundle(treatment_model_dir)?;
    let policy = V3SearchPolicy::new(&model, InferenceBackend::Neon)?;
    let mut control = LegacyTeacher::new_heuristic(load_legacy_weights(v1_weights)?, 600)?;
    let mut opponents = LegacyDirectPolicy::new(load_legacy_weights(v1_weights)?)?;
    let started = Instant::now();
    let mut records = Vec::with_capacity(pairs);
    for pair_index in first_pair_index..first_pair_index + pairs {
        let (raw_seed, seed) = protected_seed(
            seed_domain_key,
            b"final-protected-paired-k32-r600",
            pair_index,
        )?;
        let focal = pair_index % 4;
        let (treatment, treatment_seconds, treatment_overflow) =
            play_final_treatment(&policy, &mut opponents, seed, focal)?;
        let (control_score, control_seconds, control_overflow) =
            play_final_control(&mut control, &mut opponents, seed, focal)?;
        records.push(serde_json::json!({
            "pair_index": pair_index,
            "raw_seed": raw_seed,
            "focal_seat": focal,
            "treatment": {
                "score": treatment.base_total,
                "anatomy": anatomy(&treatment, treatment_overflow),
                "focal_seconds": treatment_seconds,
            },
            "control": {
                "score": control_score.base_total,
                "anatomy": anatomy(&control_score, control_overflow),
                "focal_seconds": control_seconds,
            },
            "paired_delta": i32::from(treatment.base_total) - i32::from(control_score.base_total),
            "rng_domain": "final-protected-paired-k32-r600",
            "integrity_passed": true,
        }));
    }
    write_json_atomic(
        output,
        &serde_json::json!({
            "schema_id": "cascadia-v3-final-protected-pair-shard-v1",
            "passed": true,
            "pairs": pairs,
            "first_pair_index": first_pair_index,
            "treatment_model_id": format!("{}:{}:{}", manifest.architecture_id, manifest.checkpoint_id, manifest.weights_blake3),
            "control_model_id": "qualified-exact-v1-k32-r600-f4062762",
            "records": records,
            "elapsed_seconds": started.elapsed().as_secs_f64(),
            "campaign_state_sha256": state.state_sha256,
        }),
    )?;
    Ok(())
}

fn final_all_v3(
    output: &Path,
    model_dir: &Path,
    first_game_index: usize,
    games: usize,
    seed_domain_key: &str,
    campaign_state: &Path,
    approved_readiness_sha256: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    if games == 0 || first_game_index.checked_add(games).is_none() {
        return Err("all-V3 game range is invalid".into());
    }
    let state = V3CampaignState::load_verified(campaign_state)?;
    if state.part != 2
        || state.phase != "final_all_v3_evaluation"
        || !state.phase2_authorized
        || !state.protected_seed_values_opened
        || state.approved_readiness_sha256.as_deref() != Some(approved_readiness_sha256)
        || state.readiness_sha256.as_deref() != Some(approved_readiness_sha256)
    {
        return Err("all-V3 evaluation is not open for this campaign state".into());
    }
    let (model, manifest) = QuantizedV3Model::load_bundle(model_dir)?;
    let policy = V3SearchPolicy::new(&model, InferenceBackend::Neon)?;
    let started = Instant::now();
    let mut records = Vec::with_capacity(games);
    for game_index in first_game_index..first_game_index + games {
        let (raw_seed, seed) =
            protected_seed(seed_domain_key, b"final-all-v3-k32-r600", game_index)?;
        let mut game = GameState::new(GameConfig::research_aaaaa(4)?, seed)?;
        let mut seconds = [0.0f64; 4];
        let mut overflow = [0u64; 4];
        while !game.is_game_over() {
            let seat = game.current_player();
            overflow[seat] +=
                u64::from(!encode_public_features(&game.public_state(), seat)?.natural_hot_path());
            let turn_started = Instant::now();
            let action = policy.select_action(
                &game,
                V3SearchBudget::K32R600,
                TerminalRolloutConfig {
                    model_guided: false,
                    maximum_plies: None,
                },
            )?;
            seconds[seat] += turn_started.elapsed().as_secs_f64();
            game.apply(&action)?;
        }
        let scores = score_game(&game);
        let seats = (0..4)
            .map(|seat| {
                serde_json::json!({
                    "seat": seat,
                    "score": scores[seat].base_total,
                    "anatomy": anatomy(&scores[seat], overflow[seat]),
                    "decision_seconds": seconds[seat],
                })
            })
            .collect::<Vec<_>>();
        records.push(serde_json::json!({
            "game_index": game_index,
            "raw_seed": raw_seed,
            "seats": seats,
            "rng_domain": "final-all-v3-k32-r600",
            "integrity_passed": true,
        }));
    }
    write_json_atomic(
        output,
        &serde_json::json!({
            "schema_id": "cascadia-v3-final-all-v3-shard-v1",
            "passed": true,
            "first_game_index": first_game_index,
            "games": games,
            "model_id": format!("{}:{}:{}", manifest.architecture_id, manifest.checkpoint_id, manifest.weights_blake3),
            "records": records,
            "elapsed_seconds": started.elapsed().as_secs_f64(),
            "campaign_state_sha256": state.state_sha256,
        }),
    )?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    match Args::parse().command {
        Command::Collect {
            output,
            games,
            first_game_index,
            component,
            v1_weights,
            v3_model_dir,
            cycle,
            epsilon,
            temperature,
            campaign_state,
            approved_readiness_sha256,
        } => collect(
            &output,
            games,
            first_game_index,
            component,
            v1_weights.as_deref(),
            &v3_model_dir,
            cycle,
            epsilon,
            temperature,
            campaign_state.as_deref(),
            approved_readiness_sha256.as_deref(),
        ),
        Command::VerifyShard { input, output } => verify_shard(&input, &output),
        Command::LabelRoots {
            input,
            output,
            v1_weights,
            rollouts,
            cycle,
            campaign_state,
            approved_readiness_sha256,
        } => label_roots(
            &input,
            &output,
            &v1_weights,
            rollouts,
            cycle,
            &campaign_state,
            &approved_readiness_sha256,
        ),
        Command::PromotionPairs {
            output,
            treatment_model_dir,
            control_model_dir,
            v1_weights,
            tier,
            first_pair_index,
            pairs,
            cycle,
            campaign_state,
            approved_readiness_sha256,
        } => promotion_pairs(
            &output,
            &treatment_model_dir,
            &control_model_dir,
            &v1_weights,
            tier,
            first_pair_index,
            pairs,
            cycle,
            &campaign_state,
            &approved_readiness_sha256,
        ),
        Command::FinalProtectedPairs {
            output,
            treatment_model_dir,
            v1_weights,
            first_pair_index,
            pairs,
            seed_domain_key,
            campaign_state,
            approved_readiness_sha256,
        } => final_protected_pairs(
            &output,
            &treatment_model_dir,
            &v1_weights,
            first_pair_index,
            pairs,
            &seed_domain_key,
            &campaign_state,
            &approved_readiness_sha256,
        ),
        Command::FinalAllV3 {
            output,
            model_dir,
            first_game_index,
            games,
            seed_domain_key,
            campaign_state,
            approved_readiness_sha256,
        } => final_all_v3(
            &output,
            &model_dir,
            first_game_index,
            games,
            &seed_domain_key,
            &campaign_state,
            &approved_readiness_sha256,
        ),
        Command::Health { output } => write_json_atomic(
            &output,
            &serde_json::json!({
                "schema_id": "cascadia-v3-worker-health-v1",
                "passed": true,
                "architecture": std::env::consts::ARCH,
                "cpus": std::thread::available_parallelism()?.get(),
            }),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expert_policy_rotates_one_and_only_one_newest_seat() {
        for game_index in 0..16 {
            let policies =
                choose_seat_policies(CollectionComponent::ExpertIteration, game_index, true, 4)
                    .unwrap();
            let newest = policies
                .iter()
                .enumerate()
                .filter(|(_, policy)| matches!(policy, SeatPolicy::V3(0)))
                .map(|(seat, _)| seat)
                .collect::<Vec<_>>();
            assert_eq!(newest, [game_index as usize % 4]);
        }
    }

    #[test]
    fn engineering_expert_smoke_matches_one_v3_three_v1_topology() {
        for game_index in 0..8 {
            let policies = choose_seat_policies(
                CollectionComponent::EngineeringExpertSmoke,
                game_index,
                true,
                1,
            )
            .unwrap();
            assert!(matches!(
                policies[game_index as usize % 4],
                SeatPolicy::V3(0)
            ));
            assert_eq!(
                policies
                    .iter()
                    .filter(|policy| matches!(policy, SeatPolicy::V1))
                    .count(),
                3
            );
        }
    }

    #[test]
    fn expert_opponent_pool_is_v1_dominant() {
        let mut v1 = 0usize;
        let mut prior_v3 = 0usize;
        for game_index in 0..1_000 {
            let policies =
                choose_seat_policies(CollectionComponent::ExpertIteration, game_index, true, 6)
                    .unwrap();
            for policy in policies {
                match policy {
                    SeatPolicy::V1 => v1 += 1,
                    SeatPolicy::V3(index) if index > 0 => prior_v3 += 1,
                    _ => {}
                }
            }
        }
        assert_eq!(v1 + prior_v3, 3_000);
        assert!((0.78..=0.82).contains(&(v1 as f64 / 3_000.0)));
    }

    #[test]
    fn scientific_collection_cannot_run_without_authorized_state() {
        assert!(authorize_collection(CollectionComponent::Greedy, None, None, None).is_err());
        assert!(
            authorize_collection(CollectionComponent::EngineeringSmoke, None, None, None)
                .unwrap()
                .is_none()
        );
        assert!(
            authorize_collection(
                CollectionComponent::EngineeringExpertSmoke,
                None,
                None,
                None
            )
            .unwrap()
            .is_none()
        );
    }

    #[test]
    fn promotion_tiers_bind_the_registered_search_budgets() {
        assert_eq!(
            promotion_budget(PromotionTier::Direct),
            V3SearchBudget::Direct
        );
        assert_eq!(
            promotion_budget(PromotionTier::K32R64),
            V3SearchBudget::K32R64
        );
        assert_eq!(
            promotion_budget(PromotionTier::K32R600),
            V3SearchBudget::K32R600
        );
        assert_eq!(
            promotion_budget(PromotionTier::EqualWallTime),
            V3SearchBudget::K32R600
        );
    }

    #[test]
    fn protected_seed_derivation_is_deterministic_and_domain_separated() {
        let key = "ab".repeat(32);
        let first = protected_seed(&key, b"paired", 7).unwrap().0;
        assert_eq!(first, protected_seed(&key, b"paired", 7).unwrap().0);
        assert_ne!(first, protected_seed(&key, b"paired", 8).unwrap().0);
        assert_ne!(first, protected_seed(&key, b"all-v3", 7).unwrap().0);
        assert!(protected_seed("bad", b"paired", 7).is_err());
    }
}
