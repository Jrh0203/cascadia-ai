use std::{
    fs,
    path::{Path, PathBuf},
    process::Command as ProcessCommand,
    time::Instant,
};

use cascadia_game::{GameConfig, GameSeed, GameState, Replay, score_game};
use cascadia_sim::{play_greedy_plies, select_greedy_action, strategy_rng};
use cascadia_v3_nnue::{
    InferenceBackend, QuantizedV3Model, TerminalRolloutConfig, V3AccumulatorStack, V3GameRecord,
    V3GameShardWriter, V3SearchBudget, V3SearchPolicy, V3TrainingProvenance,
    encode_public_features,
};
use clap::{Parser, Subcommand, ValueEnum};
use serde::Serialize;

#[derive(Debug, Parser)]
#[command(about = "Bounded, scientifically ineligible V3 engineering workloads")]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Profile {
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        model_dir: Option<PathBuf>,
        #[arg(long, value_enum)]
        implementation: Implementation,
        #[arg(long, default_value_t = 1)]
        replicates: usize,
        #[arg(long, default_value_t = 0)]
        greedy_plies: usize,
        #[arg(long, default_value_t = 0)]
        radial_plies: usize,
    },
    DirectGames {
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        model_dir: Option<PathBuf>,
        #[arg(long)]
        games: usize,
        #[arg(long, default_value_t = 810_000)]
        first_seed: u64,
        #[arg(long)]
        compact_out: Option<PathBuf>,
    },
    Parity {
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        model_dir: PathBuf,
        #[arg(long, default_value_t = 32)]
        states: usize,
    },
    FixtureParity {
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        model_dir: PathBuf,
        #[arg(long)]
        fixture: PathBuf,
    },
    R600Games {
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        model_dir: PathBuf,
        #[arg(long, default_value_t = 8)]
        games: usize,
        #[arg(long, default_value_t = 840_000)]
        first_seed: u64,
    },
}

#[derive(Debug, Clone, Copy, ValueEnum, Serialize)]
#[serde(rename_all = "kebab-case")]
enum Implementation {
    Reference,
    Optimized,
}

fn load_model(path: Option<&Path>) -> Result<QuantizedV3Model, Box<dyn std::error::Error>> {
    if let Some(path) = path {
        return Ok(QuantizedV3Model::load_bundle(path)?.0);
    }
    Ok(QuantizedV3Model::zeroed())
}

fn write_report(path: &Path, value: &impl Serialize) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension(format!("tmp-{}", std::process::id()));
    fs::write(&temporary, serde_json::to_vec_pretty(value)?)?;
    fs::rename(temporary, path)?;
    Ok(())
}

fn profile(
    output: &Path,
    model_dir: Option<&Path>,
    implementation: Implementation,
    replicates: usize,
    greedy_plies: usize,
    radial_plies: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    if replicates == 0 {
        return Err("replicates must be positive".into());
    }
    let model = load_model(model_dir)?;
    let policy = V3SearchPolicy::new(&model, InferenceBackend::Neon)?;
    let mut elapsed = Vec::with_capacity(replicates);
    let mut action_counts = Vec::with_capacity(replicates);
    let mut selected = Vec::with_capacity(replicates);
    let mut stage_profiles = Vec::with_capacity(replicates);
    let mut hot_path = Vec::with_capacity(replicates);
    let mut overflow_entities = Vec::with_capacity(replicates);
    for replicate in 0..replicates {
        let seed = GameSeed::from_u64(800_000 + replicate as u64);
        let mut game = GameState::new(GameConfig::research_aaaaa(4)?, seed)?;
        let mut rng = strategy_rng(seed, 0, "v3-engineering-profile-greedy-v1");
        play_greedy_plies(&mut game, greedy_plies, &mut rng)?;
        play_radial_plies(&mut game, radial_plies)?;
        let features = encode_public_features(&game.public_state(), game.current_player())?;
        hot_path.push(features.natural_hot_path());
        overflow_entities.push(
            features
                .overflow_entities
                .iter()
                .map(Vec::len)
                .sum::<usize>(),
        );
        let started = Instant::now();
        let ranked = match implementation {
            Implementation::Reference => {
                stage_profiles.push(None);
                policy.rank_legal_actions_reference(&game)?
            }
            Implementation::Optimized => {
                let (ranked, profile) = policy.rank_legal_actions_profiled(&game)?;
                stage_profiles.push(Some(profile));
                ranked
            }
        };
        elapsed.push(started.elapsed().as_secs_f64());
        action_counts.push(ranked.len());
        selected.push(
            blake3::hash(&postcard::to_allocvec(&ranked[0].action)?)
                .to_hex()
                .to_string(),
        );
    }
    let total = elapsed.iter().sum::<f64>();
    let report = serde_json::json!({
        "schema_id": "cascadia-v3-gameplay-profile-v1",
        "scientific_eligible": false,
        "implementation": implementation,
        "replicates": replicates,
        "greedy_plies": greedy_plies,
        "radial_plies": radial_plies,
        "action_counts": action_counts,
        "selected_action_blake3": selected,
        "stage_profiles": stage_profiles,
        "radius7_hot_path": hot_path,
        "overflow_entities": overflow_entities,
        "elapsed_seconds": elapsed,
        "mean_seconds_per_decision": total / replicates as f64,
        "decisions_per_second": replicates as f64 / total,
    });
    write_report(output, &report)
}

fn play_radial_plies(game: &mut GameState, plies: usize) -> Result<(), Box<dyn std::error::Error>> {
    for _ in 0..plies {
        if game.is_game_over() {
            break;
        }
        let (prelude, _) = game.preview_free_three_of_a_kind_if_feasible()?;
        let mut actions = game.legal_turn_actions(&prelude)?;
        actions.sort_by(|left, right| {
            right
                .tile
                .coord
                .radius()
                .cmp(&left.tile.coord.radius())
                .then_with(|| {
                    postcard::to_allocvec(left)
                        .expect("actions serialize")
                        .cmp(&postcard::to_allocvec(right).expect("actions serialize"))
                })
        });
        game.apply(actions.first().ok_or("radial policy found no action")?)?;
    }
    Ok(())
}

fn swap_used_bytes() -> Option<u64> {
    let output = ProcessCommand::new("/usr/sbin/sysctl")
        .args(["-n", "vm.swapusage"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8(output.stdout).ok()?;
    let fields = text.split_whitespace().collect::<Vec<_>>();
    let used = fields
        .windows(3)
        .find(|window| window[0] == "used" && window[1] == "=")?
        .get(2)?;
    let split = used.find(|character: char| !character.is_ascii_digit() && character != '.')?;
    let value = used[..split].parse::<f64>().ok()?;
    let multiplier = match &used[split..] {
        "K" => 1024.0,
        "M" => 1024.0 * 1024.0,
        "G" => 1024.0 * 1024.0 * 1024.0,
        _ => return None,
    };
    Some((value * multiplier) as u64)
}

fn direct_games(
    output: &Path,
    model_dir: Option<&Path>,
    games: usize,
    first_seed: u64,
    compact_out: Option<&Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    if games == 0 {
        return Err("games must be positive".into());
    }
    let model = load_model(model_dir)?;
    let policy = V3SearchPolicy::new(&model, InferenceBackend::Neon)?;
    let swap_before = swap_used_bytes();
    let started = Instant::now();
    let mut scores = Vec::with_capacity(games * 4);
    let mut states = 0u64;
    let mut hot_states = 0u64;
    let mut final_hashes = Vec::with_capacity(games);
    let mut compact = compact_out.map(V3GameShardWriter::create).transpose()?;
    for game_index in 0..games {
        let seed = GameSeed::from_u64(first_seed + game_index as u64);
        let mut game = GameState::new(GameConfig::research_aaaaa(4)?, seed)?;
        let mut replay = Replay::new(GameConfig::research_aaaaa(4)?, seed);
        while !game.is_game_over() {
            let features = encode_public_features(&game.public_state(), game.current_player())?;
            states += 1;
            hot_states += u64::from(features.natural_hot_path());
            let action = policy
                .rank_legal_actions(&game)?
                .into_iter()
                .next()
                .ok_or("direct V3 policy found no action")?
                .action;
            game.apply(&action)?;
            replay.turns.push(action);
        }
        replay.seal()?;
        scores.extend(score_game(&game).into_iter().map(|score| score.base_total));
        final_hashes.push(game.canonical_hash().to_hex().to_string());
        if let Some(writer) = &mut compact {
            writer.append(&V3GameRecord {
                game_index: first_seed + game_index as u64,
                replay,
                seat_policy_ids: std::array::from_fn(|_| {
                    "cascadia-v3-engineering-direct-v1".to_owned()
                }),
                newest_model_id: None,
                focal_training_seat: None,
                exploration_epsilon: 0.0,
                provenance: V3TrainingProvenance::EngineeringSmoke,
            })?;
        }
    }
    let compact_records = compact.map(V3GameShardWriter::finish).transpose()?;
    let elapsed = started.elapsed().as_secs_f64();
    let swap_after = swap_used_bytes();
    let mean = scores.iter().map(|score| f64::from(*score)).sum::<f64>() / scores.len() as f64;
    let report = serde_json::json!({
        "schema_id": "cascadia-v3-direct-game-smoke-v1",
        "scientific_eligible": false,
        "games": games,
        "seat_scores": scores,
        "mean_base_score": mean,
        "states": states,
        "hot_path_states": hot_states,
        "hot_path_fraction": hot_states as f64 / states as f64,
        "overflow_states": states - hot_states,
        "elapsed_seconds": elapsed,
        "seconds_per_game": elapsed / games as f64,
        "games_per_hour": games as f64 * 3600.0 / elapsed,
        "swap_before_bytes": swap_before,
        "swap_after_bytes": swap_after,
        "swap_delta_bytes": match (swap_before, swap_after) {
            (Some(before), Some(after)) => after.saturating_sub(before),
            _ => u64::MAX,
        },
        "final_state_blake3": final_hashes,
        "compact_shard": compact_out.map(|path| serde_json::json!({
            "path": path,
            "games": compact_records,
            "bytes": path.metadata().map(|metadata| metadata.len()).unwrap_or(0),
        })),
    });
    write_report(output, &report)
}

fn r600_games(
    output: &Path,
    model_dir: &Path,
    games: usize,
    first_seed: u64,
) -> Result<(), Box<dyn std::error::Error>> {
    if games == 0 {
        return Err("games must be positive".into());
    }
    let model = load_model(Some(model_dir))?;
    let policy = V3SearchPolicy::new(&model, InferenceBackend::Neon)?;
    let swap_before = swap_used_bytes();
    let started = Instant::now();
    let mut seat_scores = Vec::with_capacity(games * 4);
    let mut focal_scores = Vec::with_capacity(games);
    let mut decision_seconds = Vec::with_capacity(games * 20);
    let mut final_hashes = Vec::with_capacity(games);
    for game_index in 0..games {
        let seed = GameSeed::from_u64(first_seed + game_index as u64);
        let focal = game_index % 4;
        let mut game = GameState::new(GameConfig::research_aaaaa(4)?, seed)?;
        let mut rngs = (0..4)
            .map(|seat| strategy_rng(seed, seat, "v3-r600-smoke-greedy-opponents-v1"))
            .collect::<Vec<_>>();
        while !game.is_game_over() {
            let seat = game.current_player();
            let action = if seat == focal {
                let decision_started = Instant::now();
                let action = policy.select_action(
                    &game,
                    V3SearchBudget::K32R600,
                    TerminalRolloutConfig {
                        model_guided: false,
                        maximum_plies: None,
                    },
                )?;
                decision_seconds.push(decision_started.elapsed().as_secs_f64());
                action
            } else {
                let (prelude, _) = game.preview_free_three_of_a_kind_if_feasible()?;
                select_greedy_action(&game, &prelude, &mut rngs[seat])?
            };
            game.apply(&action)?;
        }
        let scores = score_game(&game)
            .into_iter()
            .map(|score| score.base_total)
            .collect::<Vec<_>>();
        focal_scores.push(scores[focal]);
        seat_scores.extend(scores);
        final_hashes.push(game.canonical_hash().to_hex().to_string());
    }
    let elapsed = started.elapsed().as_secs_f64();
    let swap_after = swap_used_bytes();
    write_report(
        output,
        &serde_json::json!({
            "schema_id": "cascadia-v3-r600-game-smoke-v1",
            "scientific_eligible": false,
            "games": games,
            "first_seed": first_seed,
            "focal_seat_rule": "game-index-mod-4",
            "rollout_policy": "terminal-greedy",
            "search_budget": {"candidates": 32, "rollouts": 600},
            "focal_scores": focal_scores,
            "seat_scores": seat_scores,
            "r600_decisions": decision_seconds.len(),
            "decision_seconds": decision_seconds,
            "elapsed_seconds": elapsed,
            "r600_seconds_per_game": elapsed / games as f64,
            "swap_before_bytes": swap_before,
            "swap_after_bytes": swap_after,
            "swap_delta_bytes": match (swap_before, swap_after) {
                (Some(before), Some(after)) => after.saturating_sub(before),
                _ => u64::MAX,
            },
            "final_state_blake3": final_hashes,
        }),
    )
}

fn parity(
    output: &Path,
    model_dir: &Path,
    states: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let model = load_model(Some(model_dir))?;
    let mut checked = 0usize;
    let mut values = Vec::new();
    for game_index in 0..states {
        let seed = GameSeed::from_u64(820_000 + game_index as u64);
        let game = GameState::new(GameConfig::research_aaaaa(4)?, seed)?;
        let features = encode_public_features(&game.public_state(), game.current_player())?;
        let scalar = V3AccumulatorStack::new(&model, features.clone(), InferenceBackend::Scalar)?;
        let neon = V3AccumulatorStack::new(&model, features, InferenceBackend::Neon)?;
        let scalar_value = scalar.evaluate(&model)?;
        let neon_value = neon.evaluate(&model)?;
        if scalar_value != neon_value || scalar != neon {
            return Err(format!("scalar/NEON mismatch at state {game_index}").into());
        }
        values.push(scalar_value.raw_output_units);
        checked += 1;
    }
    let report = serde_json::json!({
        "schema_id": "cascadia-v3-rust-backend-parity-v1",
        "scientific_eligible": false,
        "states": checked,
        "bit_identical": true,
        "raw_output_units": values,
    });
    write_report(output, &report)
}

#[derive(Debug, serde::Deserialize)]
struct ParityFixture {
    schema_id: String,
    rows: Vec<ParityFixtureRow>,
}

#[derive(Debug, serde::Deserialize)]
struct ParityFixtureRow {
    features: cascadia_v3_nnue::V3FeatureSet,
    expected_raw_output_units: i32,
}

fn fixture_parity(
    output: &Path,
    model_dir: &Path,
    fixture: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let model = load_model(Some(model_dir))?;
    let fixture: ParityFixture = serde_json::from_reader(fs::File::open(fixture)?)?;
    if fixture.schema_id != "cascadia-v3-quantized-parity-fixture-v1" || fixture.rows.is_empty() {
        return Err("invalid V3 parity fixture".into());
    }
    let mut values = Vec::with_capacity(fixture.rows.len());
    for (index, row) in fixture.rows.into_iter().enumerate() {
        let scalar =
            V3AccumulatorStack::new(&model, row.features.clone(), InferenceBackend::Scalar)?;
        let neon = V3AccumulatorStack::new(&model, row.features, InferenceBackend::Neon)?;
        let scalar_value = scalar.evaluate(&model)?;
        let neon_value = neon.evaluate(&model)?;
        if scalar_value != neon_value
            || scalar_value.raw_output_units != row.expected_raw_output_units
        {
            let trace = scalar.trace(&model)?;
            write_report(
                output,
                &serde_json::json!({
                    "schema_id": "cascadia-v3-cross-backend-parity-failure-v1",
                    "row": index,
                    "expected_raw_output_units": row.expected_raw_output_units,
                    "scalar": scalar_value,
                    "neon": neon_value,
                    "own_accumulator": scalar.own_accumulator(),
                    "field_accumulator": scalar.field_accumulator(),
                    "trace": trace,
                }),
            )?;
            return Err(format!(
                "quantized parity mismatch at fixture row {index}: expected {}, scalar {}, neon {}; diagnostics written to {}",
                row.expected_raw_output_units,
                scalar_value.raw_output_units,
                neon_value.raw_output_units,
                output.display(),
            )
            .into());
        }
        values.push(scalar_value.raw_output_units);
    }
    write_report(
        output,
        &serde_json::json!({
            "schema_id": "cascadia-v3-cross-backend-parity-v1",
            "rows": values.len(),
            "rust_scalar_neon_bit_identical": true,
            "rust_mlx_quantized_bit_identical": true,
            "raw_output_units": values,
        }),
    )
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    match args.command {
        Command::Profile {
            output,
            model_dir,
            implementation,
            replicates,
            greedy_plies,
            radial_plies,
        } => profile(
            &output,
            model_dir.as_deref(),
            implementation,
            replicates,
            greedy_plies,
            radial_plies,
        ),
        Command::DirectGames {
            output,
            model_dir,
            games,
            first_seed,
            compact_out,
        } => direct_games(
            &output,
            model_dir.as_deref(),
            games,
            first_seed,
            compact_out.as_deref(),
        ),
        Command::Parity {
            output,
            model_dir,
            states,
        } => parity(&output, &model_dir, states),
        Command::FixtureParity {
            output,
            model_dir,
            fixture,
        } => fixture_parity(&output, &model_dir, &fixture),
        Command::R600Games {
            output,
            model_dir,
            games,
            first_seed,
        } => r600_games(&output, &model_dir, games, first_seed),
    }
}
