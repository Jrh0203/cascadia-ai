use std::{
    error::Error,
    fs::{self, File},
    io::{self, BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
};

use cascadia_r2::{
    CorpusRequirement, DatasetIdentity, SparsePublicState, SuppliedTile, census_datasets,
    export_mlx_cache, read_record_at_ordinal, write_json_atomic,
};
use clap::{Parser, Subcommand};
use serde::Serialize;

#[derive(Debug, Parser)]
#[command(
    name = "r2-sparse-entity-census",
    about = "Exact sparse occupied-plus-frontier tokenization and corpus census"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Validate and census one or more compact-entity-v2 dataset roots.
    Census {
        #[arg(long = "dataset-root", required = true)]
        dataset_roots: Vec<PathBuf>,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        require_r0_corpus: bool,
        #[arg(long, value_name = "A,B_OR_NONE,MASK,KEYSTONE")]
        supplied_tile: Option<SuppliedTile>,
    },
    /// Emit one fully expanded token state by global corpus ordinal.
    Tokenize {
        #[arg(long = "dataset-root", required = true)]
        dataset_roots: Vec<PathBuf>,
        #[arg(long)]
        ordinal: usize,
        #[arg(long)]
        output: Option<PathBuf>,
        #[arg(long)]
        packed_output: Option<PathBuf>,
        #[arg(long, value_name = "A,B_OR_NONE,MASK,KEYSTONE")]
        supplied_tile: Option<SuppliedTile>,
    },
    /// Decode and validate one canonical CSR2SP1 packed state.
    Decode {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Export the frozen exact R2 token corpus as one content-addressed MLX cache.
    ExportMlx {
        #[arg(long)]
        corpus_lock: PathBuf,
        #[arg(long = "dataset-root", required = true)]
        dataset_roots: Vec<PathBuf>,
        #[arg(long)]
        output_root: PathBuf,
        #[arg(long)]
        receipt: Option<PathBuf>,
    },
}

#[derive(Debug, Serialize)]
struct TokenizeOutput {
    ordinal: usize,
    dataset: DatasetIdentity,
    packed_bytes: usize,
    packed_blake3: String,
    state: SparsePublicState,
}

#[derive(Debug, Serialize)]
struct DecodeOutput {
    packed_bytes: usize,
    packed_blake3: String,
    state: SparsePublicState,
}

fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();
    match cli.command {
        Command::Census {
            dataset_roots,
            output,
            require_r0_corpus,
            supplied_tile,
        } => {
            let requirement = if require_r0_corpus {
                CorpusRequirement::AcceptedR0SixtyThousand
            } else {
                CorpusRequirement::AnyValidatedCompactEntityV2
            };
            let report = census_datasets(&dataset_roots, supplied_tile, requirement)?;
            write_json_atomic(&output, &report)?;
            println!(
                "{} records; scientific BLAKE3 {}",
                report.scientific.record_count, report.scientific_blake3
            );
        }
        Command::Tokenize {
            dataset_roots,
            ordinal,
            output,
            packed_output,
            supplied_tile,
        } => {
            let (dataset, record) = read_record_at_ordinal(&dataset_roots, ordinal)?;
            let state = SparsePublicState::from_position_record(&record, supplied_tile)?;
            let packed = state.to_packed_bytes()?;
            let result = TokenizeOutput {
                ordinal,
                dataset,
                packed_bytes: packed.len(),
                packed_blake3: blake3::hash(&packed).to_hex().to_string(),
                state,
            };
            write_json_or_stdout(output.as_deref(), &result)?;
            if let Some(path) = packed_output {
                write_bytes_atomic(&path, &packed)?;
            }
        }
        Command::Decode { input, output } => {
            let mut bytes = Vec::new();
            BufReader::new(File::open(input)?).read_to_end(&mut bytes)?;
            let state = SparsePublicState::from_packed_bytes(&bytes)?;
            let result = DecodeOutput {
                packed_bytes: bytes.len(),
                packed_blake3: blake3::hash(&bytes).to_hex().to_string(),
                state,
            };
            write_json_or_stdout(output.as_deref(), &result)?;
        }
        Command::ExportMlx {
            corpus_lock,
            dataset_roots,
            output_root,
            receipt,
        } => {
            let result = export_mlx_cache(
                &corpus_lock,
                &dataset_roots,
                &output_root,
                receipt.as_deref(),
            )?;
            write_json_or_stdout(None, &result)?;
        }
    }
    Ok(())
}

fn write_json_or_stdout(path: Option<&Path>, value: &impl Serialize) -> Result<(), Box<dyn Error>> {
    if let Some(path) = path {
        write_json_atomic(path, value)?;
    } else {
        let mut stdout = io::stdout().lock();
        serde_json::to_writer_pretty(&mut stdout, value)?;
        stdout.write_all(b"\n")?;
    }
    Ok(())
}

fn write_bytes_atomic(path: &Path, bytes: &[u8]) -> Result<(), Box<dyn Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temp = path.with_extension("bin.tmp");
    let mut writer = BufWriter::new(File::create(&temp)?);
    writer.write_all(bytes)?;
    writer.flush()?;
    writer.get_ref().sync_all()?;
    fs::rename(temp, path)?;
    Ok(())
}
