use std::{
    fs,
    io::Read,
    path::{Path, PathBuf},
    time::Instant,
};

use cascadia_v3_nnue::{
    V3LabeledTeacherRootShardReader, V3TrainingShardWriter, labeled_root_training_entries,
};
use clap::Parser;

#[derive(Debug, Parser)]
#[command(about = "Expand labeled V3 teacher roots into counterfactual sparse training rows")]
struct Args {
    #[arg(long, required = true)]
    input: Vec<PathBuf>,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    receipt: PathBuf,
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    if let Some(parent) = args.output.parent() {
        fs::create_dir_all(parent)?;
    }
    let started = Instant::now();
    let mut writer = V3TrainingShardWriter::create(&args.output)?;
    let mut roots = 0u64;
    let mut counterfactual_rows = 0u64;
    let mut realized_rows = 0u64;
    for input in &args.input {
        let mut reader = V3LabeledTeacherRootShardReader::open(input)?;
        while let Some(value) = reader.next_labeled_root()? {
            let realized_action =
                &value.root.record.replay.turns[usize::from(value.root.turn_index)];
            for (candidate, entry) in value
                .label
                .candidates
                .iter()
                .zip(labeled_root_training_entries(&value)?)
            {
                if &candidate.action == realized_action {
                    realized_rows += 1;
                } else {
                    counterfactual_rows += 1;
                }
                writer.append(&entry)?;
            }
            roots += 1;
        }
    }
    let rows = writer.finish()?;
    if rows != counterfactual_rows + realized_rows {
        return Err("teacher training-row accounting differs".into());
    }
    let receipt = serde_json::json!({
        "schema_id": "cascadia-v3-teacher-training-expansion-v1",
        "passed": true,
        "scientific_eligible": true,
        "inputs": args.input.iter().map(|path| serde_json::json!({
            "path": path,
            "bytes": path.metadata().map(|value| value.len()).unwrap_or(0),
            "blake3": checksum(path).unwrap_or_default(),
        })).collect::<Vec<_>>(),
        "roots": roots,
        "rows": rows,
        "realized_rows": realized_rows,
        "counterfactual_rows": counterfactual_rows,
        "output": args.output,
        "output_bytes": args.output.metadata()?.len(),
        "output_blake3": checksum(&args.output)?,
        "elapsed_seconds": started.elapsed().as_secs_f64(),
    });
    if let Some(parent) = args.receipt.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = args
        .receipt
        .with_extension(format!("tmp-{}", std::process::id()));
    fs::write(&temporary, serde_json::to_vec_pretty(&receipt)?)?;
    fs::rename(temporary, &args.receipt)?;
    println!("{}", serde_json::to_string(&receipt)?);
    Ok(())
}
