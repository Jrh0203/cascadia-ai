use std::{
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

use cascadia_game::{GameConfig, GameSeed, GameState, score_board, score_game};
use cascadia_sim::{select_greedy_action, strategy_rng};
use cascadia_v3_nnue::{
    V3TrainingEntry, V3TrainingProvenance, V3TrainingShardWriter, encode_public_features,
    signed_score_to_go,
};
use clap::Parser;
use rayon::prelude::*;
use serde::Serialize;

#[derive(Debug, Parser)]
#[command(about = "Collect the bounded, non-scientific V3 engineering corpus")]
struct Args {
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = 2_000)]
    games: usize,
    #[arg(long, default_value_t = 100)]
    games_per_shard: usize,
    #[arg(long, default_value_t = 700_000)]
    first_seed: u64,
}

#[derive(Debug, Serialize)]
struct ShardReceipt {
    file: String,
    first_game_index: u64,
    games: usize,
    records: u64,
    bytes: u64,
    blake3: String,
    states: u64,
    hot_path_states: u64,
    active_opportunity_rows_sum: u64,
    active_opportunity_rows_min: usize,
    active_opportunity_rows_max: usize,
    own_opportunity_rows_sum: u64,
    field_opportunity_rows_sum: u64,
}

#[derive(Debug)]
struct PendingEntry {
    state_blake3: [u8; 32],
    decision_index: u8,
    focal_seat: u8,
    features: cascadia_v3_nnue::V3FeatureSet,
    current_score: u16,
}

fn checksum(path: &Path) -> Result<String, std::io::Error> {
    use std::io::Read;
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

fn collect_shard(
    output: &Path,
    shard_index: usize,
    first_seed: u64,
    games: usize,
) -> Result<ShardReceipt, Box<dyn std::error::Error + Send + Sync>> {
    let path = output.join(format!("engineering-{shard_index:05}.v3t"));
    let mut writer = V3TrainingShardWriter::create(&path)?;
    let mut states = 0u64;
    let mut hot_path_states = 0u64;
    let mut active_opportunity_rows_sum = 0u64;
    let mut active_opportunity_rows_min = usize::MAX;
    let mut active_opportunity_rows_max = 0usize;
    let mut own_opportunity_rows_sum = 0u64;
    let mut field_opportunity_rows_sum = 0u64;
    for game_offset in 0..games {
        let game_index = first_seed + game_offset as u64;
        let seed = GameSeed::from_u64(game_index);
        let config = GameConfig::research_aaaaa(4)?;
        let mut game = GameState::new(config, seed)?;
        let mut rngs = (0..4)
            .map(|seat| strategy_rng(seed, seat, "v3-engineering-greedy-v1"))
            .collect::<Vec<_>>();
        let mut pending = Vec::with_capacity(80);
        while !game.is_game_over() {
            let focal = game.current_player();
            let decision_index = game.boards()[focal].tile_count().saturating_sub(3) as u8;
            let (prelude, _) = game.preview_free_three_of_a_kind_if_feasible()?;
            let action = select_greedy_action(&game, &prelude, &mut rngs[focal])?;
            game.apply(&action)?;
            let public = game.public_state();
            let features = encode_public_features(&public, focal)?;
            states += 1;
            hot_path_states += u64::from(features.natural_hot_path());
            let active_opportunities =
                features.own_opportunities.len() + features.field_opportunities.len();
            active_opportunity_rows_sum += active_opportunities as u64;
            active_opportunity_rows_min = active_opportunity_rows_min.min(active_opportunities);
            active_opportunity_rows_max = active_opportunity_rows_max.max(active_opportunities);
            own_opportunity_rows_sum += features.own_opportunities.len() as u64;
            field_opportunity_rows_sum += features.field_opportunities.len() as u64;
            pending.push(PendingEntry {
                state_blake3: *public.canonical_hash().as_bytes(),
                decision_index,
                focal_seat: focal as u8,
                features,
                current_score: score_board(&game.boards()[focal], config.scoring_cards).base_total,
            });
        }
        let final_scores = score_game(&game);
        for pending in pending {
            let score_to_go = signed_score_to_go(
                final_scores[usize::from(pending.focal_seat)].base_total,
                pending.current_score,
            );
            writer.append(&V3TrainingEntry {
                state_blake3: pending.state_blake3,
                game_index,
                decision_index: pending.decision_index,
                focal_seat: pending.focal_seat,
                features: pending.features,
                realized_score_to_go: score_to_go,
                teacher_score_to_go: None,
                teacher_variance: None,
                teacher_sample_count: 0,
                lambda: 0.0,
                target_score_to_go: score_to_go,
                provenance: V3TrainingProvenance::EngineeringSmoke,
            })?;
        }
    }
    let records = writer.finish()?;
    Ok(ShardReceipt {
        file: path.file_name().unwrap().to_string_lossy().into_owned(),
        first_game_index: first_seed,
        games,
        records,
        bytes: path.metadata()?.len(),
        blake3: checksum(&path)?,
        states,
        hot_path_states,
        active_opportunity_rows_sum,
        active_opportunity_rows_min,
        active_opportunity_rows_max,
        own_opportunity_rows_sum,
        field_opportunity_rows_sum,
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    if args.games == 0 || args.games_per_shard == 0 {
        return Err("games and games-per-shard must be positive".into());
    }
    fs::create_dir_all(&args.output)?;
    let started = Instant::now();
    let shards = args.games.div_ceil(args.games_per_shard);
    let receipts = (0..shards)
        .into_par_iter()
        .map(|shard| {
            let game_offset = shard * args.games_per_shard;
            let games = args.games_per_shard.min(args.games - game_offset);
            collect_shard(
                &args.output,
                shard,
                args.first_seed + game_offset as u64,
                games,
            )
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| -> Box<dyn std::error::Error> { error })?;
    let states = receipts.iter().map(|receipt| receipt.states).sum::<u64>();
    let hot = receipts
        .iter()
        .map(|receipt| receipt.hot_path_states)
        .sum::<u64>();
    let records = receipts.iter().map(|receipt| receipt.records).sum::<u64>();
    let active_opportunity_rows_sum = receipts
        .iter()
        .map(|receipt| receipt.active_opportunity_rows_sum)
        .sum::<u64>();
    let active_opportunity_rows_min = receipts
        .iter()
        .map(|receipt| receipt.active_opportunity_rows_min)
        .min()
        .unwrap_or(0);
    let active_opportunity_rows_max = receipts
        .iter()
        .map(|receipt| receipt.active_opportunity_rows_max)
        .max()
        .unwrap_or(0);
    let own_opportunity_rows_sum = receipts
        .iter()
        .map(|receipt| receipt.own_opportunity_rows_sum)
        .sum::<u64>();
    let field_opportunity_rows_sum = receipts
        .iter()
        .map(|receipt| receipt.field_opportunity_rows_sum)
        .sum::<u64>();
    let manifest = serde_json::json!({
        "schema_id": "cascadia-v3-engineering-corpus-v1",
        "dataset_class": "engineering_smoke",
        "scientific_eligible": false,
        "first_seed": args.first_seed,
        "games": args.games,
        "records": records,
        "expected_records": args.games * 80,
        "states": states,
        "hot_path_states": hot,
        "hot_path_fraction": hot as f64 / states as f64,
        "overflow_states": states - hot,
        "active_opportunity_rows": {
            "mean": active_opportunity_rows_sum as f64 / states as f64,
            "min": active_opportunity_rows_min,
            "max": active_opportunity_rows_max,
            "own_mean": own_opportunity_rows_sum as f64 / states as f64,
            "three_opponent_field_mean": field_opportunity_rows_sum as f64 / states as f64,
            "per_board_perspective_mean": active_opportunity_rows_sum as f64
                / states as f64
                / 4.0,
        },
        "elapsed_seconds": started.elapsed().as_secs_f64(),
        "shards": receipts,
    });
    let path = args.output.join("dataset.json");
    let temporary = args.output.join(".dataset.json.tmp");
    fs::write(&temporary, serde_json::to_vec_pretty(&manifest)?)?;
    fs::rename(temporary, path)?;
    println!("{}", serde_json::to_string(&manifest)?);
    Ok(())
}
