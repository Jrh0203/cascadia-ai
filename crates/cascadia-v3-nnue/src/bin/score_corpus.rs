use std::{fs, path::PathBuf};

use cascadia_game::score_game;
use cascadia_v3_nnue::V3GameShardReader;
use clap::{Parser, ValueEnum};
use serde::Serialize;

#[derive(Debug, Clone, Copy, ValueEnum)]
enum FocalMode {
    RecordFocal,
    GameIndexSeat,
}

#[derive(Debug, Parser)]
#[command(about = "Score focal seats from compact V3 replay shards")]
struct Args {
    #[arg(long, required = true)]
    input: Vec<PathBuf>,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, value_enum)]
    focal_mode: FocalMode,
}

#[derive(Debug, Serialize)]
struct ScoreRow {
    game_index: u64,
    focal_seat: u8,
    score: u16,
    wildlife: [u16; 5],
    habitat: [u16; 5],
    nature_tokens: u16,
    pinecones: u16,
}

fn percentile(sorted: &[u16], quantile: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let position = quantile * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    let fraction = position - lower as f64;
    f64::from(sorted[lower]) * (1.0 - fraction) + f64::from(sorted[upper]) * fraction
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let mut rows = Vec::new();
    for input in &args.input {
        let mut reader = V3GameShardReader::open(input)?;
        while let Some(record) = reader.next_record()? {
            let focal = match args.focal_mode {
                FocalMode::RecordFocal => record
                    .focal_training_seat
                    .ok_or("record has no registered focal training seat")?,
                FocalMode::GameIndexSeat => (record.game_index % 4) as u8,
            };
            let terminal = record.replay.play()?;
            let score = score_game(&terminal)[usize::from(focal)];
            rows.push(ScoreRow {
                game_index: record.game_index,
                focal_seat: focal,
                score: score.base_total,
                wildlife: score.wildlife,
                habitat: score.habitat,
                nature_tokens: score.nature_tokens,
                pinecones: score.nature_tokens,
            });
        }
    }
    if rows.is_empty() {
        return Err("score corpus is empty".into());
    }
    rows.sort_by_key(|row| row.game_index);
    if rows
        .windows(2)
        .any(|pair| pair[0].game_index == pair[1].game_index)
    {
        return Err("score corpus contains duplicate game indices".into());
    }
    let mut scores = rows.iter().map(|row| row.score).collect::<Vec<_>>();
    scores.sort_unstable();
    let count = rows.len() as f64;
    let mean = rows.iter().map(|row| f64::from(row.score)).sum::<f64>() / count;
    let variance = if rows.len() > 1 {
        rows.iter()
            .map(|row| (f64::from(row.score) - mean).powi(2))
            .sum::<f64>()
            / (count - 1.0)
    } else {
        0.0
    };
    let sum_array = |select: fn(&ScoreRow) -> [u16; 5]| -> [u64; 5] {
        let mut totals = [0u64; 5];
        for row in &rows {
            for (target, value) in totals.iter_mut().zip(select(row)) {
                *target += u64::from(value);
            }
        }
        totals
    };
    let result = serde_json::json!({
        "schema_id": "cascadia-v3-focal-score-corpus-v1",
        "passed": true,
        "games": rows.len(),
        "mean": mean,
        "standard_error": (variance / count).sqrt(),
        "p10": percentile(&scores, 0.10),
        "p50": percentile(&scores, 0.50),
        "p90": percentile(&scores, 0.90),
        "wildlife_totals": sum_array(|row| row.wildlife),
        "habitat_totals": sum_array(|row| row.habitat),
        "nature_token_total": rows.iter().map(|row| u64::from(row.nature_tokens)).sum::<u64>(),
        "pinecone_total": rows.iter().map(|row| u64::from(row.pinecones)).sum::<u64>(),
        "rows": rows,
    });
    if let Some(parent) = args.output.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = args
        .output
        .with_extension(format!("tmp-{}", std::process::id()));
    fs::write(&temporary, serde_json::to_vec_pretty(&result)?)?;
    fs::rename(temporary, &args.output)?;
    println!(
        "{}",
        serde_json::to_string(&serde_json::json!({
            "games": result["games"],
            "mean": result["mean"],
            "p10": result["p10"],
            "p90": result["p90"],
        }))?
    );
    Ok(())
}
