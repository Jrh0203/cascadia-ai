use std::{fs, path::PathBuf};

use cascadia_v3_nnue::V3FeatureSchemaManifest;
use clap::Parser;

#[derive(Debug, Parser)]
struct Args {
    #[arg(long)]
    output: Option<PathBuf>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let manifest = V3FeatureSchemaManifest::build()?;
    let rendered = serde_json::to_string_pretty(&manifest)? + "\n";
    if let Some(path) = args.output {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, rendered)?;
    } else {
        print!("{rendered}");
    }
    Ok(())
}
