use std::{
    fs,
    io::Read,
    path::{Path, PathBuf},
};

use cascadia_v3_nnue::{V3TeacherRootShardReader, V3TeacherRootShardWriter};
use clap::Parser;
use serde::Serialize;

#[derive(Debug, Parser)]
#[command(about = "Split a verified V3 teacher-root corpus into scheduler-sized shards")]
struct Args {
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    output_dir: PathBuf,
    #[arg(long)]
    prefix: String,
    #[arg(long, default_value_t = 1_000)]
    roots_per_shard: usize,
    #[arg(long)]
    receipt: PathBuf,
}

#[derive(Debug, Serialize)]
struct ShardReceipt {
    file: String,
    roots: u64,
    bytes: u64,
    blake3: String,
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
    if args.prefix.is_empty()
        || !args
            .prefix
            .chars()
            .all(|value| value.is_ascii_alphanumeric() || value == '-')
        || args.roots_per_shard == 0
    {
        return Err("prefix or roots-per-shard is invalid".into());
    }
    fs::create_dir_all(&args.output_dir)?;
    let mut reader = V3TeacherRootShardReader::open(&args.input)?;
    let expected = reader.len();
    let mut shards = Vec::new();
    let mut total = 0u64;
    let mut shard_index = 0usize;
    loop {
        let path = args
            .output_dir
            .join(format!("{}-{shard_index:05}.v3r", args.prefix));
        let mut writer = V3TeacherRootShardWriter::create(&path)?;
        let mut count = 0u64;
        while count < args.roots_per_shard as u64 {
            let Some(root) = reader.next_root()? else {
                break;
            };
            writer.append(&root)?;
            count += 1;
        }
        if count == 0 {
            fs::remove_file(path)?;
            break;
        }
        let written = writer.finish()?;
        if written != count {
            return Err("teacher root shard count changed".into());
        }
        total += count;
        shards.push(ShardReceipt {
            file: path.file_name().unwrap().to_string_lossy().into_owned(),
            roots: count,
            bytes: path.metadata()?.len(),
            blake3: checksum(&path)?,
        });
        shard_index += 1;
    }
    if total != expected {
        return Err("split root count differs from input".into());
    }
    let receipt = serde_json::json!({
        "schema_id": "cascadia-v3-teacher-root-split-v1",
        "passed": true,
        "scientific_eligible": true,
        "input": args.input,
        "input_bytes": args.input.metadata()?.len(),
        "input_blake3": checksum(&args.input)?,
        "roots": total,
        "roots_per_shard": args.roots_per_shard,
        "shards": shards,
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
