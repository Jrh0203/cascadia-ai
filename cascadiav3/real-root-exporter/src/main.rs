#![recursion_limit = "256"]

use std::collections::{BTreeMap, HashMap, HashSet};
mod feature_tensors;
mod gumbel;
mod model_bridge;
mod npz_writer;

use std::fs::File;
use std::io::{BufRead, BufWriter, Read, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use cascadia_game::{
    DraftChoice, GameConfig, GameSeed, GameState, HexCoord, MarketPrelude, MarketSlot, RuleError,
    ScoreBreakdown, Tile, TurnAction, Wildlife, score_game,
};
use cascadia_sim::{GreedyCandidate, SimulationError, rank_greedy_actions};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

use feature_tensors::{
    EXPERT_SHARD_VERSION, EXPERT_SHARD_VERSION_V2, ExpertTensorShardData,
    PUBLIC_TOKEN_FEATURE_DIM, SEMANTIC_PUBLIC_TOKEN_ACTION_FEATURE_DIM, SHARD_VERSION,
    TensorShardData,
};
use model_bridge::{
    BridgeConfig, ModelEval, ModelServiceSession, SharedBridge, SharedBridgeClient,
    uniform_model_eval,
};
use npz_writer::NpzCompression;

const SCHEMA_ID: &str = "cascadiav3.pre_gpu.v0";
const EXPERT_ROOT_SCHEMA_ID: &str = "cascadiav3.expert_root.v1";
const EXPERT_TENSOR_SCHEMA_ID: &str = "cascadiav3.expert_tensor_shard.v1";
const EXPERT_TENSOR_SCHEMA_ID_V2: &str = "cascadiav3.expert_tensor_shard.v2";
const GREEDY_TENSOR_SCHEMA_ID: &str = "greedy_policy_tensor_shard_v1";
const ROOT_REPLAY_SCHEMA_ID: &str = "cascadiav3.root_replay.v1";
const RULESET_ID: &str = "cascadia_research_aaaaa_4p_card_a_no_habitat_bonus";
const DEFAULT_FIRST_SEED: u64 = 2_026_062_900;
const DEFAULT_SEED_COUNT: u64 = 2;
const DEFAULT_PLIES_PER_SEED: usize = 2;
const DEFAULT_MAX_ACTIONS: usize = 8;
const DEFAULT_ROLLOUTS_PER_ACTION: usize = 1;
const DEFAULT_ROLLOUT_TOP_K: usize = 1;
const SOFTMAX_TEMPERATURE: f64 = 10.0;
const HIDDEN_DETERMINIZATION_SALT: u64 = 0x5eed_c0de_d371_a11e;

#[derive(Debug, Clone)]
struct Args {
    mode: Mode,
    out: PathBuf,
    manifest: PathBuf,
    first_seed: u64,
    seed_count: u64,
    plies_per_seed: usize,
    max_actions: usize,
    rollouts_per_action: usize,
    rollout_top_k: usize,
    player_count: u8,
    rayon_threads: Option<usize>,
    tensor_compression: NpzCompression,
    input: Option<PathBuf>,
    bench_out: Option<PathBuf>,
    allow_model_fallback: bool,
    model_service: Option<String>,
    model_manifest: Option<PathBuf>,
    model_timeout_ms: u64,
    rollout_determinize: bool,
    gumbel_n_simulations: usize,
    gumbel_top_m: usize,
    gumbel_depth_rounds: usize,
    gumbel_determinizations: usize,
    gumbel_blend_weight: f64,
    gumbel_exploration: bool,
    gumbel_max_root_actions: Option<usize>,
    /// Root menu enumeration cap (immediate-score-ranked pre-filter before
    /// the model-prior top-m). 0 = full legal set. Late-game legal menus can
    /// exceed several thousand compound actions, which both bloats eval
    /// requests and blows up the relation-bias memory (B x A x S x d).
    gumbel_root_menu: usize,
    k_interior: usize,
    model_sessions: Option<usize>,
    /// One shared bridge (one CUDA context) with cross-chunk request
    /// batching instead of one bridge per rayon chunk.
    shared_model_session: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    ExportRoots,
    ChanceMctsDryRun,
    ValidateExpertReconstruction,
    ValidateHiddenRedetermination,
    BenchTokenize,
    GreedyPolicyCorpus,
    GreedyPolicyTensorCorpus,
    GreedyExpertTensorCorpus,
    GreedyStateSearchBootstrapTensorCorpus,
    ModelStateSearchBootstrapTensorCorpus,
    ExpertTensorCorpus,
    InteractivePolicyGame,
    GumbelPolicyGame,
    GumbelSelfplayTensorCorpus,
}

#[derive(Debug, Clone)]
struct ActionRollout {
    candidate: GreedyCandidate,
    active_score: f64,
    active_score_variance: f64,
    sample_count: usize,
    truncated_count: usize,
    score_means: Vec<ScoreMean>,
}

#[derive(Debug, Clone)]
struct ExpertActionEval {
    action: TurnAction,
    action_id: String,
    action_token: Value,
    afterstate_hash: String,
    afterstate_public_hash: String,
    exact_afterstate_score_active: f64,
    active_score: f64,
    active_score_variance: f64,
    sample_count: usize,
    truncated_count: usize,
    rollout_seeds: Vec<u64>,
    score_means: Vec<ScoreMean>,
}

#[derive(Debug, Clone)]
struct ExpertRootBuild {
    record: Value,
    prelude: MarketPrelude,
    staged: GameState,
    selected_action: TurnAction,
    selected_action_id: String,
}

#[derive(Debug, Clone, Copy)]
struct ScoreMean {
    wildlife: f64,
    habitat: f64,
    nature_tokens: f64,
    total: f64,
}

fn main() -> Result<()> {
    let args = parse_args()?;
    match args.mode {
        Mode::ExportRoots => {
            let records = export_records(&args)?;
            if records.is_empty() {
                bail!("exporter produced no records");
            }
            write_jsonl(&args.out, &records)?;
            write_manifest(&args.manifest, &records, &args)?;
            println!(
                "wrote {} simulator roots to {} and manifest {}",
                records.len(),
                args.out.display(),
                args.manifest.display()
            );
        }
        Mode::ChanceMctsDryRun => {
            let started = std::time::Instant::now();
            let records = export_expert_records(&args)?;
            if records.is_empty() {
                bail!("expert dry-run exporter produced no records");
            }
            write_jsonl(&args.out, &records)?;
            write_expert_manifest(&args.manifest, &records, &args)?;
            if let Some(path) = &args.bench_out {
                write_expert_search_bench(path, &records, started.elapsed().as_secs_f64())?;
            }
            println!(
                "wrote {} expert dry-run roots to {} and manifest {}",
                records.len(),
                args.out.display(),
                args.manifest.display()
            );
        }
        Mode::ValidateExpertReconstruction => {
            let report = validate_expert_reconstruction(&args)?;
            println!("{}", canonical_json(&report));
        }
        Mode::ValidateHiddenRedetermination => {
            let report = validate_hidden_redetermination(&args)?;
            if let Some(parent) = args.out.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&args.out, format!("{}\n", canonical_json(&report)))?;
            println!("{}", canonical_json(&report));
        }
        Mode::BenchTokenize => {
            let report = bench_tokenize(&args)?;
            if let Some(parent) = args.out.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&args.out, format!("{}\n", canonical_json(&report)))?;
            println!("{}", canonical_json(&report));
        }
        Mode::GreedyPolicyCorpus => {
            let written = export_greedy_policy_corpus(&args)?;
            if written == 0 {
                bail!("greedy policy corpus exporter produced no records");
            }
            eprintln!(
                "wrote {} greedy policy roots to {} and manifest {}",
                written,
                args.out.display(),
                args.manifest.display()
            );
        }
        Mode::GreedyPolicyTensorCorpus => {
            let written = export_greedy_policy_tensor_corpus(&args)?;
            if written == 0 {
                bail!("greedy policy tensor exporter produced no records");
            }
            eprintln!(
                "wrote {} greedy policy tensor roots to {} and manifest {}",
                written,
                args.out.display(),
                args.manifest.display()
            );
        }
        Mode::GreedyExpertTensorCorpus => {
            let written = export_greedy_expert_tensor_corpus(&args)?;
            if written == 0 {
                bail!("greedy expert tensor exporter produced no records");
            }
            eprintln!(
                "wrote {} greedy expert tensor roots to {} and manifest {}",
                written,
                args.out.display(),
                args.manifest.display()
            );
        }
        Mode::GreedyStateSearchBootstrapTensorCorpus => {
            let written = export_greedy_state_search_bootstrap_tensor_corpus(&args)?;
            if written == 0 {
                bail!("greedy-state search-bootstrap tensor exporter produced no records");
            }
            eprintln!(
                "wrote {} greedy-state search-bootstrap tensor roots to {} and manifest {}",
                written,
                args.out.display(),
                args.manifest.display()
            );
        }
        Mode::ModelStateSearchBootstrapTensorCorpus => {
            let written = export_model_state_search_bootstrap_tensor_corpus(&args)?;
            if written == 0 {
                bail!("model-state search-bootstrap tensor exporter produced no records");
            }
            eprintln!(
                "wrote {} model-state search-bootstrap tensor roots to {} and manifest {}",
                written,
                args.out.display(),
                args.manifest.display()
            );
        }
        Mode::ExpertTensorCorpus => {
            let written = export_expert_tensor_corpus(&args)?;
            if written == 0 {
                bail!("expert tensor exporter produced no records");
            }
            eprintln!(
                "wrote {} expert tensor roots to {} and manifest {}",
                written,
                args.out.display(),
                args.manifest.display()
            );
        }
        Mode::InteractivePolicyGame => {
            run_interactive_policy_game(&args)?;
        }
        Mode::GumbelPolicyGame => {
            run_gumbel_policy_game(&args)?;
        }
        Mode::GumbelSelfplayTensorCorpus => {
            let written = export_gumbel_selfplay_tensor_corpus(&args)?;
            if written == 0 {
                bail!("gumbel selfplay tensor exporter produced no records");
            }
            eprintln!(
                "wrote {} gumbel selfplay tensor roots to {} and manifest {}",
                written,
                args.out.display(),
                args.manifest.display()
            );
        }
    }
    Ok(())
}

fn parse_args() -> Result<Args> {
    let mut args = Args {
        mode: Mode::ExportRoots,
        out: PathBuf::from("cascadiav3/fixtures/real_roots.jsonl"),
        manifest: PathBuf::from("cascadiav3/fixtures/real_roots_manifest.json"),
        first_seed: DEFAULT_FIRST_SEED,
        seed_count: DEFAULT_SEED_COUNT,
        plies_per_seed: DEFAULT_PLIES_PER_SEED,
        max_actions: DEFAULT_MAX_ACTIONS,
        rollouts_per_action: DEFAULT_ROLLOUTS_PER_ACTION,
        rollout_top_k: DEFAULT_ROLLOUT_TOP_K,
        player_count: 4,
        rayon_threads: None,
        tensor_compression: NpzCompression::Deflate,
        input: None,
        bench_out: None,
        allow_model_fallback: false,
        model_service: None,
        model_manifest: None,
        model_timeout_ms: 10_000,
        rollout_determinize: false,
        gumbel_n_simulations: 64,
        gumbel_top_m: 16,
        gumbel_depth_rounds: 1,
        gumbel_determinizations: 4,
        gumbel_blend_weight: 0.5,
        gumbel_exploration: false,
        gumbel_max_root_actions: None,
        gumbel_root_menu: 256,
        k_interior: 16,
        model_sessions: None,
        shared_model_session: false,
    };

    let mut iter = std::env::args().skip(1);
    while let Some(flag) = iter.next() {
        let mut value = || {
            iter.next()
                .with_context(|| format!("{flag} requires a value"))
        };
        match flag.as_str() {
            "--chance-mcts-dry-run" => args.mode = Mode::ChanceMctsDryRun,
            "--validate-expert-reconstruction" => args.mode = Mode::ValidateExpertReconstruction,
            "--validate-hidden-redetermination" => args.mode = Mode::ValidateHiddenRedetermination,
            "--bench-tokenize" => args.mode = Mode::BenchTokenize,
            "--interactive-policy-game" => args.mode = Mode::InteractivePolicyGame,
            "--greedy-policy-corpus" => args.mode = Mode::GreedyPolicyCorpus,
            "--greedy-policy-tensor-corpus" => args.mode = Mode::GreedyPolicyTensorCorpus,
            "--greedy-expert-tensor-corpus" => args.mode = Mode::GreedyExpertTensorCorpus,
            "--greedy-state-search-bootstrap-tensor-corpus" => {
                args.mode = Mode::GreedyStateSearchBootstrapTensorCorpus
            }
            "--model-state-search-bootstrap-tensor-corpus" => {
                args.mode = Mode::ModelStateSearchBootstrapTensorCorpus
            }
            "--expert-tensor-corpus" => args.mode = Mode::ExpertTensorCorpus,
            "--gumbel-policy-game" => args.mode = Mode::GumbelPolicyGame,
            "--gumbel-selfplay-tensor-corpus" => {
                args.mode = Mode::GumbelSelfplayTensorCorpus;
                args.gumbel_exploration = true;
            }
            "--gumbel-n-simulations" => {
                args.gumbel_n_simulations =
                    value()?.parse().context("invalid --gumbel-n-simulations")?
            }
            "--gumbel-top-m" => {
                args.gumbel_top_m = value()?.parse().context("invalid --gumbel-top-m")?
            }
            "--gumbel-depth-rounds" => {
                args.gumbel_depth_rounds =
                    value()?.parse().context("invalid --gumbel-depth-rounds")?
            }
            "--gumbel-determinizations" => {
                args.gumbel_determinizations = value()?
                    .parse()
                    .context("invalid --gumbel-determinizations")?
            }
            "--gumbel-blend-weight" => {
                args.gumbel_blend_weight =
                    value()?.parse().context("invalid --gumbel-blend-weight")?
            }
            "--gumbel-exploration" => {
                args.gumbel_exploration = match value()?.as_str() {
                    "on" | "true" | "1" => true,
                    "off" | "false" | "0" => false,
                    other => bail!("invalid --gumbel-exploration {other}; use on|off"),
                }
            }
            "--gumbel-root-menu" => {
                args.gumbel_root_menu =
                    value()?.parse().context("invalid --gumbel-root-menu")?
            }
            "--gumbel-max-root-actions" => {
                args.gumbel_max_root_actions = Some(
                    value()?
                        .parse()
                        .context("invalid --gumbel-max-root-actions")?,
                )
            }
            "--k-interior" => {
                args.k_interior = value()?.parse().context("invalid --k-interior")?
            }
            "--model-sessions" => {
                args.model_sessions =
                    Some(value()?.parse().context("invalid --model-sessions")?)
            }
            "--shared-model-session" => args.shared_model_session = true,
            "--out" => args.out = PathBuf::from(value()?),
            "--in" => args.input = Some(PathBuf::from(value()?)),
            "--manifest" => args.manifest = PathBuf::from(value()?),
            "--bench-out" => args.bench_out = Some(PathBuf::from(value()?)),
            "--allow-model-fallback" => args.allow_model_fallback = true,
            "--rollout-determinize" => args.rollout_determinize = true,
            "--model-service" => args.model_service = Some(value()?),
            "--model-manifest" => args.model_manifest = Some(PathBuf::from(value()?)),
            "--model-timeout-ms" => {
                args.model_timeout_ms = value()?.parse().context("invalid --model-timeout-ms")?
            }
            "--first-seed" => args.first_seed = value()?.parse().context("invalid --first-seed")?,
            "--seed-count" => args.seed_count = value()?.parse().context("invalid --seed-count")?,
            "--plies-per-seed" => {
                args.plies_per_seed = value()?.parse().context("invalid --plies-per-seed")?;
            }
            "--max-actions" => {
                args.max_actions = value()?.parse().context("invalid --max-actions")?
            }
            "--rollouts-per-action" => {
                args.rollouts_per_action =
                    value()?.parse().context("invalid --rollouts-per-action")?
            }
            "--rollout-top-k" => {
                args.rollout_top_k = value()?.parse().context("invalid --rollout-top-k")?
            }
            "--player-count" => {
                args.player_count = value()?.parse().context("invalid --player-count")?
            }
            "--rayon-threads" => {
                args.rayon_threads = Some(value()?.parse().context("invalid --rayon-threads")?)
            }
            "--tensor-compression" => {
                args.tensor_compression = parse_tensor_compression(&value()?)?
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            _ => bail!("unknown argument {flag}; use --help"),
        }
    }

    if args.seed_count == 0 {
        bail!("--seed-count must be positive");
    }
    if args.plies_per_seed == 0 {
        bail!("--plies-per-seed must be positive");
    }
    if args.max_actions == 0 {
        bail!("--max-actions must be positive");
    }
    if matches!(
        args.mode,
        Mode::ExportRoots
            | Mode::ChanceMctsDryRun
            | Mode::ExpertTensorCorpus
            | Mode::GreedyStateSearchBootstrapTensorCorpus
            | Mode::ModelStateSearchBootstrapTensorCorpus
    ) && args.rollouts_per_action == 0
    {
        bail!("--rollouts-per-action must be positive");
    }
    if args.rollout_top_k == 0 {
        bail!("--rollout-top-k must be positive");
    }
    if args.player_count != 4 {
        bail!("v3 schema currently requires 4-player score vectors");
    }
    if args.mode == Mode::ValidateExpertReconstruction && args.input.is_none() {
        bail!("--validate-expert-reconstruction requires --in <expert_roots.jsonl>");
    }
    if matches!(
        args.mode,
        Mode::ChanceMctsDryRun
            | Mode::ExpertTensorCorpus
            | Mode::ModelStateSearchBootstrapTensorCorpus
            | Mode::GumbelPolicyGame
            | Mode::GumbelSelfplayTensorCorpus
    ) && args.model_service.is_none()
        && !args.allow_model_fallback
    {
        bail!(
            "expert export requires --model-service for model priors, or --allow-model-fallback for dry-run uniform priors"
        );
    }
    if matches!(
        args.mode,
        Mode::GumbelPolicyGame | Mode::GumbelSelfplayTensorCorpus
    ) {
        if args.gumbel_n_simulations == 0 {
            bail!("--gumbel-n-simulations must be positive");
        }
        if args.gumbel_top_m == 0 {
            bail!("--gumbel-top-m must be positive");
        }
        if !(0.0..=1.0).contains(&args.gumbel_blend_weight) {
            bail!("--gumbel-blend-weight must be in [0, 1]");
        }
        if args.k_interior == 0 {
            bail!("--k-interior must be positive");
        }
    }
    if args.model_service.is_some() && args.model_manifest.is_none() && !args.allow_model_fallback {
        bail!(
            "real model service use requires --model-manifest unless --allow-model-fallback is set"
        );
    }
    if let Some(threads) = args.rayon_threads {
        if threads == 0 {
            bail!("--rayon-threads must be positive");
        }
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .context("initializing Rayon global thread pool")?;
    }
    Ok(args)
}

fn parse_tensor_compression(raw: &str) -> Result<NpzCompression> {
    match raw {
        "deflate" => Ok(NpzCompression::Deflate),
        "stored" | "none" | "uncompressed" => Ok(NpzCompression::Stored),
        _ => bail!("--tensor-compression must be deflate or stored"),
    }
}

fn tensor_compression_label(compression: NpzCompression) -> &'static str {
    match compression {
        NpzCompression::Deflate => "deflate",
        NpzCompression::Stored => "stored",
    }
}

fn print_help() {
    println!(
        "\
cascadiav3-real-root-exporter

Generate dry-run v3 search-root JSONL from the canonical Cascadia simulator.

Options:
  --chance-mcts-dry-run
                           Export expert-root v1 records with full legal
                           coverage and dry-run rollout labels. Requires
                           --model-service or --allow-model-fallback.
  --allow-model-fallback   Allow uniform-prior dry-run fallback when no model
                           service is attached.
  --model-service <cmd>    Python JSONL stdio model service command for real
                           model priors. Real use requires --model-manifest.
  --model-manifest <path>  Manifest checked before model service use.
  --model-timeout-ms <n>   Model service request timeout [10000].
  --validate-expert-reconstruction
                           Rebuild expert roots from seed plus replay prefix
                           and verify legal action order and hashes.
  --validate-hidden-redetermination
                           Validate that hidden redetermination preserves
                           public state, legal actions, and public supply.
  --bench-tokenize         Write tokenization/action enumeration microbench JSON.
  --in <path>              Input expert JSONL for validation modes.
  --bench-out <path>       Optional dry-run search benchmark JSON output.
  --out <path>             JSONL output path, or '-' for stdout
                           [cascadiav3/fixtures/real_roots.jsonl]
  --manifest <path>        Manifest output path [cascadiav3/fixtures/real_roots_manifest.json]
  --first-seed <u64>       First deterministic simulator seed [{DEFAULT_FIRST_SEED}]
  --seed-count <u64>       Number of consecutive seeds [{DEFAULT_SEED_COUNT}]
  --plies-per-seed <n>     Number of selected roots to advance per seed [{DEFAULT_PLIES_PER_SEED}]
  --max-actions <n>        Greedy-ranked legal actions retained per root [{DEFAULT_MAX_ACTIONS}]
  --rollouts-per-action <n>
                           Terminal rollout samples per retained action [{DEFAULT_ROLLOUTS_PER_ACTION}]
  --rollout-top-k <n>      Sample each continuation from top-k greedy actions [{DEFAULT_ROLLOUT_TOP_K}]
  --rollout-determinize    Resample hidden stack/bag order per rollout so
                           search and labels never observe the true hidden
                           draw order (public-information-legal search).
  --gumbel-policy-game     Play seeds to terminal with all four seats driven
                           by Gumbel search over model leaf values; emits
                           per-decision JSONL plus a done record per seed.
  --gumbel-selfplay-tensor-corpus
                           All-seat Gumbel self-play; every visited root is
                           exported as a v2 expert tensor record with
                           completed-Q targets, improved policy, and real
                           final-outcome value labels.
  --gumbel-n-simulations <n>
                           Total simulation budget per decision [64].
  --gumbel-top-m <n>       Gumbel top-m root candidates [16].
  --gumbel-depth-rounds <n>
                           Root-seat re-entries before leaf valuation [1].
  --gumbel-determinizations <n>
                           Hidden-order determinizations cycled per action [4].
  --gumbel-blend-weight <w>
                           Leaf value = w*model bootstrap + (1-w)*greedy
                           rollout [0.5].
  --gumbel-exploration <on|off>
                           Gumbel exploration noise at the root [off; selfplay
                           mode defaults on].
  --gumbel-max-root-actions <n>
                           Optional model-prior-ranked cap on root candidates.
  --gumbel-root-menu <n>   Root menu enumeration cap before model ranking;
                           0 keeps the full legal set [256].
  --k-interior <n>         Interior-ply menu cap inside simulations [16].
  --model-sessions <n>     Cap on concurrent model bridge sessions for
                           selfplay export chunks (with --shared-model-session
                           this is just the parallel game count).
  --shared-model-session   One bridge process (one CUDA context) serving all
                           chunks with cross-chunk request batching.
  --player-count <n>       Must be 4 for the current v3 schema [4]
  --rayon-threads <n>      Set the Rayon worker thread count explicitly
  --tensor-compression <deflate|stored>
                           Compression for --greedy-policy-tensor-corpus NPZ
                           arrays [deflate]
  --interactive-policy-game
                           Stream policy roots on stdout, read JSON decisions
                           on stdin, and play one complete game from
                           --first-seed. This is for v3 model/search pilots.
  --greedy-policy-corpus
                           Export complete greedy self-play roots with no
                           per-action search/rollouts for behavior-cloning
                           pretraining.
  --greedy-policy-tensor-corpus
                           Export the same greedy self-play roots as compact
                           trainer-ready float16 tensor .npz shards, using the
                           Rust port of the public-token/semantic-action
                           feature schema.
  --greedy-expert-tensor-corpus
                           Export greedy self-play roots as packed
                           cascadiav3.expert_tensor_shard.v1 .npz, including
                           relation edges and schema-valid auxiliary labels for
                           CascadiaFormer greedy-retention training.
  --greedy-state-search-bootstrap-tensor-corpus
                           Export greedy self-play state roots as packed
                           cascadiav3.expert_tensor_shard.v1 .npz, evaluate
                           the retained greedy-ranked action menu with sampled
                           greedy rollouts, label the rollout-best action, and
                           still advance each real trajectory with the greedy
                           action to avoid first-cycle distribution shift.
  --model-state-search-bootstrap-tensor-corpus
                           Export model-visited state roots as packed
                           cascadiav3.expert_tensor_shard.v1 .npz. The rollout
                           teacher still labels the best retained action, while
                           the real trajectory advances by model derived-Q when
                           present, otherwise by model prior. Requires
                           --model-service or --allow-model-fallback.
  --expert-tensor-corpus   Export expert-root training data directly as packed
                           cascadiav3.expert_tensor_shard.v1 .npz without
                           writing JSONL. This is the scale path.
"
    );
}

fn export_records(args: &Args) -> Result<Vec<Value>> {
    let mut records = Vec::new();
    let mut per_seed = (0..args.seed_count)
        .into_par_iter()
        .map(|offset| {
            let seed_u64 = args.first_seed + offset;
            let result = export_seed_records(args, seed_u64)
                .with_context(|| format!("exporting seed {seed_u64}"));
            (offset, result)
        })
        .collect::<Vec<_>>();
    per_seed.sort_by_key(|(offset, _)| *offset);
    for (_, seed_records) in per_seed {
        records.extend(seed_records?);
    }
    Ok(records)
}

fn export_expert_records(args: &Args) -> Result<Vec<Value>> {
    let mut records = Vec::new();
    let mut per_seed = (0..args.seed_count)
        .into_par_iter()
        .map(|offset| {
            let seed_u64 = args.first_seed + offset;
            let result = export_expert_seed_records(args, seed_u64)
                .with_context(|| format!("exporting expert seed {seed_u64}"));
            (offset, result)
        })
        .collect::<Vec<_>>();
    per_seed.sort_by_key(|(offset, _)| *offset);
    for (_, seed_records) in per_seed {
        records.extend(seed_records?);
    }
    Ok(records)
}

fn export_expert_seed_records(args: &Args, seed_u64: u64) -> Result<Vec<Value>> {
    let config = GameConfig::research_aaaaa(args.player_count)?;
    let mut game = GameState::new(config, GameSeed::from_u64(seed_u64))
        .with_context(|| format!("creating expert seed {seed_u64}"))?;
    let mut records = Vec::new();
    let mut replay_prefix = Vec::new();
    for ply_index in 0..args.plies_per_seed {
        if game.is_game_over() {
            break;
        }
        let root = build_expert_root_record(&game, seed_u64, ply_index, &replay_prefix, args)
            .with_context(|| format!("expert seed {seed_u64} ply {ply_index}"))?;
        let before_public_hash = public_hash(&root.staged);
        let mut next = root.staged.clone();
        next.apply(&root.selected_action)
            .context("applying selected expert action")?;
        let after_public_hash = public_hash(&next);
        replay_prefix.push(json!({
            "schema_id": "cascadiav3.root_replay_step.v1",
            "ply": ply_index,
            "prelude": root.prelude,
            "action": root.selected_action,
            "action_id": root.selected_action_id,
            "before_public_hash": before_public_hash,
            "after_public_hash": after_public_hash,
            "after_full_hash": game_hash(&next),
        }));
        records.push(root.record);
        game = next;
    }
    Ok(records)
}

fn build_expert_root_record(
    game: &GameState,
    seed_u64: u64,
    ply_index: usize,
    replay_prefix: &[Value],
    args: &Args,
) -> Result<ExpertRootBuild> {
    let active_seat = game.current_player();
    let parent_public_hash = public_hash(game);
    let parent_full_hash = game_hash(game);
    let (prelude, staged) = game.preview_free_three_of_a_kind_if_feasible()?;
    let staged_public_hash = public_hash(&staged);
    let staged_full_hash = game_hash(&staged);
    let root_public_hash = staged_public_hash.clone();
    let root_full_hash = staged_full_hash.clone();
    let legal = staged
        .legal_turn_actions(&MarketPrelude::default())
        .context("enumerating full expert legal action set")?;
    if legal.is_empty() {
        bail!("no legal actions for expert root");
    }

    let mut evals = Vec::with_capacity(legal.len());
    for (action_index, action) in legal.iter().enumerate() {
        let action_id = action_id(action)?;
        let mut after = staged.clone();
        after
            .apply(action)
            .with_context(|| format!("applying expert root action {action_index}"))?;
        let afterstate_hash = game_hash(&after);
        let afterstate_public_hash = public_hash(&after);
        let after_scores = score_game(&after);
        let exact_afterstate_score_active = f64::from(after_scores[active_seat].total);

        let mut score_samples = Vec::with_capacity(args.rollouts_per_action);
        let mut active_samples = Vec::with_capacity(args.rollouts_per_action);
        let mut rollout_seeds = Vec::with_capacity(args.rollouts_per_action);
        let mut truncated_count = 0usize;
        for rollout_index in 0..args.rollouts_per_action {
            let rollout_seed = rollout_seed(seed_u64, ply_index, action_index, rollout_index);
            rollout_seeds.push(rollout_seed);
            let mut rng = ChaCha8Rng::seed_from_u64(rollout_seed);
            let (terminal, truncated) = if args.rollout_determinize {
                let (sim, apply_truncated) =
                    determinized_afterstate(&staged, action, rollout_seed)?;
                if apply_truncated {
                    (sim, true)
                } else {
                    complete_with_sampled_greedy(
                        sim,
                        args.max_actions,
                        args.rollout_top_k,
                        &mut rng,
                        None,
                    )?
                }
            } else {
                complete_with_sampled_greedy(
                    after.clone(),
                    args.max_actions,
                    args.rollout_top_k,
                    &mut rng,
                    None,
                )?
            };
            truncated_count += usize::from(truncated);
            let terminal_scores = score_game(&terminal);
            active_samples.push(f64::from(terminal_scores[active_seat].total));
            score_samples.push(terminal_scores);
        }
        let (active_score, active_score_variance) = mean_variance(&active_samples);
        let score_means = mean_scores(&score_samples);
        let action_token = action_token(
            &staged,
            &prelude,
            action,
            after_scores[active_seat].base_total,
            active_seat,
            action_index,
        )?;
        evals.push(ExpertActionEval {
            action: action.clone(),
            action_id,
            action_token,
            afterstate_hash,
            afterstate_public_hash,
            exact_afterstate_score_active,
            active_score,
            active_score_variance,
            sample_count: active_samples.len(),
            truncated_count,
            rollout_seeds,
            score_means,
        });
    }

    let selected_index = evals
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| {
            left.active_score
                .partial_cmp(&right.active_score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| right.action_id.cmp(&left.action_id))
        })
        .map(|(index, _)| index)
        .context("expert root has no selectable action")?;
    let selected = &evals[selected_index];
    let selected_scores = &selected.score_means;
    let action_count = evals.len();
    let final_score_vector = selected_scores
        .iter()
        .map(|score| json!(score.total))
        .collect::<Vec<_>>();
    let score_decomposition = score_decomposition(selected_scores);
    let rank_vector = rank_vector(selected_scores)
        .into_iter()
        .map(|rank| json!(rank))
        .collect::<Vec<_>>();
    let q_values = evals
        .iter()
        .map(|eval| json!(eval.active_score))
        .collect::<Vec<_>>();
    let afterstate_scores = evals
        .iter()
        .map(|eval| json!(eval.exact_afterstate_score_active))
        .collect::<Vec<_>>();
    let score_to_go = evals
        .iter()
        .map(|eval| json!(eval.active_score - eval.exact_afterstate_score_active))
        .collect::<Vec<_>>();
    let action_ids = evals
        .iter()
        .map(|eval| json!(eval.action_id))
        .collect::<Vec<_>>();
    let afterstate_hashes = evals
        .iter()
        .map(|eval| json!(eval.afterstate_hash))
        .collect::<Vec<_>>();
    let afterstate_public_hashes = evals
        .iter()
        .map(|eval| json!(eval.afterstate_public_hash))
        .collect::<Vec<_>>();
    let legal_actions = evals
        .iter()
        .map(|eval| eval.action_token.clone())
        .collect::<Vec<_>>();
    let model_request = json!({
        "schema_id": EXPERT_ROOT_SCHEMA_ID,
        "ruleset_id": RULESET_ID,
        "state_hash": root_public_hash,
        "public_hash": root_public_hash,
        "seed": seed_u64,
        "ply": ply_index,
        "active_seat": active_seat,
        "legal_actions": legal_actions.clone(),
        "action_ids": action_ids.clone(),
        "exact_afterstate_score_active": afterstate_scores.clone(),
        "public_tokens": public_tokens(&staged, active_seat),
    });
    let model_eval = model_eval_for_root(args, &model_request, action_count)
        .with_context(|| format!("model-service eval for seed {seed_u64} ply {ply_index}"))?;
    let visits = evals
        .iter()
        .map(|eval| json!(eval.sample_count))
        .collect::<Vec<_>>();
    let per_action_q_variance = evals
        .iter()
        .map(|eval| json!(eval.active_score_variance))
        .collect::<Vec<_>>();
    let per_action_q_count = evals
        .iter()
        .map(|eval| json!(eval.sample_count))
        .collect::<Vec<_>>();
    let per_action_truncated_count = evals
        .iter()
        .map(|eval| json!(eval.truncated_count))
        .collect::<Vec<_>>();
    let per_action_rollout_seeds = evals
        .iter()
        .map(|eval| {
            Value::Array(
                eval.rollout_seeds
                    .iter()
                    .map(|seed| json!(seed))
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<Vec<_>>();
    let selected_rollout_seed = rollout_seed(seed_u64, ply_index, selected_index, 0);
    let selected_chance_samples = selected
        .rollout_seeds
        .iter()
        .enumerate()
        .map(|(rollout_index, rollout_seed)| {
            json!({
                "sample_id": format!("chance:{}:{}:{}:{}", seed_u64, ply_index, selected_index, rollout_index),
                "action_id": selected.action_id,
                "seed": rollout_seed,
                "probability": 1.0 / args.rollouts_per_action as f64,
                "logprob": (1.0 / args.rollouts_per_action as f64).ln(),
                "before_hash": staged_full_hash,
                "after_hash": selected.afterstate_hash,
                "before_public_hash": staged_public_hash,
                "after_public_hash": selected.afterstate_public_hash,
                "public_delta": {
                    "public_hash_changed": staged_public_hash != selected.afterstate_public_hash,
                    "active_seat": active_seat,
                },
                "private_audit_hash": selected.afterstate_hash,
            })
        })
        .collect::<Vec<_>>();

    let mut record = json!({
        "schema_id": EXPERT_ROOT_SCHEMA_ID,
        "ruleset_id": RULESET_ID,
        "state_hash": root_public_hash,
        "public_hash": root_public_hash,
        "seed": seed_u64,
        "ply": ply_index,
        "active_seat": active_seat,
        "source_hash": format!("sha256:{}", sha256_hex(include_str!("main.rs").as_bytes())),
        "binary_hash": binary_hash(),
        "root_replay": {
            "schema_id": ROOT_REPLAY_SCHEMA_ID,
            "config_id": RULESET_ID,
            "seed_u64": seed_u64,
            "ply": ply_index,
            "replay_prefix": replay_prefix,
            "prefix_action_count": replay_prefix.len(),
            "market_prelude": prelude,
            "parent_full_hash": parent_full_hash,
            "parent_public_hash": parent_public_hash,
            "root_full_hash": root_full_hash,
            "root_public_hash": root_public_hash,
            "staged_full_hash": staged_full_hash,
            "staged_public_hash": staged_public_hash,
        },
        "actor_identity": {
            "kind": "expert_iteration_dry_run_actor",
            "seat": active_seat,
            "policy": "chance_mcts_dry_run_best_q",
        },
        "opponent_identities": opponent_identities(active_seat),
        "model_identity": {
            "kind": if model_eval.model_fallback { "uniform_prior_fallback" } else { "python_jsonl_stdio" },
            "model_fallback": model_eval.model_fallback,
            "service": args.model_service.as_ref(),
            "manifest": args.model_manifest.as_ref().map(|path| path.display().to_string()),
            "timeout_ms": args.model_timeout_ms,
            "response": model_eval.response,
        },
        "search_identity": {
            "kind": "chance_mcts_dry_run",
            "primary_utility": "active_seat_raw_final_score",
            "simulations_per_action": args.rollouts_per_action,
            "rollout_policy": "sampled_greedy",
            "rollout_top_k": args.rollout_top_k,
            "continuation_max_actions": args.max_actions,
            "full_legal_coverage": true,
        },
        "rng_identity": {
            "root_seed_u64": seed_u64,
            "rollout_seed_domain": "rollout_seed(seed,ply,action_index,rollout_index)",
            "chance_sample_seed": selected_rollout_seed,
        },
        "legal_actions": legal_actions,
        "action_ids": action_ids,
        "afterstate_hashes": afterstate_hashes,
        "afterstate_public_hashes": afterstate_public_hashes,
        "exact_afterstate_score_active": afterstate_scores,
        "priors": model_eval.priors,
        "visits": visits,
        "per_action_Q": q_values,
        "per_action_score_to_go": score_to_go,
        "per_action_Q_valid": vec![json!(true); action_count],
        "per_action_Q_variance": per_action_q_variance,
        "per_action_Q_count": per_action_q_count,
        "per_action_truncated_count": per_action_truncated_count,
        "per_action_rollout_seeds": per_action_rollout_seeds,
        "selected_action": selected.action_id,
        "chance_samples": selected_chance_samples,
        "final_score_vector": final_score_vector,
        "score_decomposition": score_decomposition,
        "rank_vector": rank_vector,
        "public_tokens": public_tokens(&staged, active_seat),
        "metadata": {
            "source": "canonical_simulator_chance_mcts_dry_run_expert_root",
            "scientific_eligibility": "dry_run",
            "root_seed_u64": seed_u64,
            "ply_index_for_seed": ply_index,
            "completed_turns": game.completed_turns(),
            "turns_remaining": game.turns_remaining(),
            "full_legal_action_count": action_count,
            "legal_action_coverage": 1.0,
            "q_label_semantics": "active_seat_final_raw_score",
            "score_to_go_semantics": "per_action_Q - exact_afterstate_score_active",
            "q_valid_invalid_policy": "legal_unvisited_actions_present_with_per_action_Q_valid_false; dry_run_visits_all_actions_once",
            "model_fallback": model_eval.model_fallback,
            "rollouts_per_action": args.rollouts_per_action,
            "prelude_replace_three_of_a_kind": prelude.replace_three_of_a_kind,
            "prelude_wildlife_wipe_count": prelude.wildlife_wipes.len(),
        },
    });
    attach_checksum(&mut record)?;
    Ok(ExpertRootBuild {
        record,
        prelude,
        staged,
        selected_action: selected.action.clone(),
        selected_action_id: selected.action_id.clone(),
    })
}

fn opponent_identities(active_seat: usize) -> Vec<Value> {
    (0..4)
        .filter(|seat| *seat != active_seat)
        .map(|seat| {
            json!({
                "seat": seat,
                "relative_seat": relative_seat(active_seat, seat),
                "policy": "sampled_greedy_rollout_control",
            })
        })
        .collect()
}

fn model_eval_for_root(
    args: &Args,
    root_request: &Value,
    action_count: usize,
) -> Result<ModelEval> {
    if let Some(command) = &args.model_service {
        match call_model_service(command, root_request, args) {
            Ok(eval) => return Ok(eval),
            Err(error) if args.allow_model_fallback => {
                let mut eval = uniform_model_eval(action_count);
                eval.response = json!({
                    "type": "eval_response",
                    "model_fallback": true,
                    "fallback_reason": error.to_string(),
                });
                return Ok(eval);
            }
            Err(error) => return Err(error),
        }
    }
    if args.allow_model_fallback {
        return Ok(uniform_model_eval(action_count));
    }
    bail!("expert export requires --model-service or explicit --allow-model-fallback")
}

fn call_model_service(command: &str, root_request: &Value, args: &Args) -> Result<ModelEval> {
    let mut session = ModelServiceSession::spawn(command, &BridgeConfig::from_args(args))?;
    let eval = session.eval(root_request)?;
    session.shutdown();
    Ok(eval)
}

fn validate_expert_reconstruction(args: &Args) -> Result<Value> {
    let input = args
        .input
        .as_ref()
        .context("--validate-expert-reconstruction requires --in")?;
    let records = read_jsonl(input)?;
    let manifest: Value = serde_json::from_str(
        &std::fs::read_to_string(&args.manifest)
            .with_context(|| format!("reading manifest {}", args.manifest.display()))?,
    )
    .with_context(|| format!("parsing manifest {}", args.manifest.display()))?;
    if manifest.get("schema_id").and_then(Value::as_str) != Some(EXPERT_ROOT_SCHEMA_ID) {
        bail!("expert manifest schema_id mismatch");
    }
    if manifest.get("record_count").and_then(Value::as_u64) != Some(records.len() as u64) {
        bail!("expert manifest record_count does not match input records");
    }

    let mut legal_actions_checked = 0usize;
    let mut selected_actions_checked = 0usize;
    for (record_index, record) in records.iter().enumerate() {
        validate_one_expert_reconstruction(record)
            .with_context(|| format!("validating expert reconstruction record {record_index}"))?;
        legal_actions_checked += record
            .get("legal_actions")
            .and_then(Value::as_array)
            .map(Vec::len)
            .unwrap_or(0);
        selected_actions_checked += 1;
    }
    Ok(json!({
        "status": "pass",
        "records": records.len(),
        "legal_actions_checked": legal_actions_checked,
        "selected_actions_checked": selected_actions_checked,
        "input": input.display().to_string(),
        "manifest": args.manifest.display().to_string(),
    }))
}

fn validate_one_expert_reconstruction(record: &Value) -> Result<()> {
    if record.get("schema_id").and_then(Value::as_str) != Some(EXPERT_ROOT_SCHEMA_ID) {
        bail!("record schema_id is not {EXPERT_ROOT_SCHEMA_ID}");
    }
    verify_record_checksum(record)?;
    let seed_u64 = record
        .get("seed")
        .and_then(Value::as_u64)
        .context("expert record missing seed")?;
    let config = GameConfig::research_aaaaa(4)?;
    let mut game = GameState::new(config, GameSeed::from_u64(seed_u64))?;
    let root_replay = record
        .get("root_replay")
        .and_then(Value::as_object)
        .context("expert record missing root_replay")?;
    let prefix = root_replay
        .get("replay_prefix")
        .and_then(Value::as_array)
        .context("root_replay.replay_prefix missing")?;
    for (step_index, step) in prefix.iter().enumerate() {
        let prelude: MarketPrelude = serde_json::from_value(
            step.get("prelude")
                .cloned()
                .with_context(|| format!("prefix step {step_index} missing prelude"))?,
        )?;
        let action: TurnAction = serde_json::from_value(
            step.get("action")
                .cloned()
                .with_context(|| format!("prefix step {step_index} missing action"))?,
        )?;
        let staged = game.preview_market_prelude(&prelude)?;
        game = staged;
        game.apply(&action)
            .with_context(|| format!("applying prefix step {step_index}"))?;
        if let Some(expected) = step.get("after_full_hash").and_then(Value::as_str) {
            let actual = game_hash(&game);
            if actual != expected {
                bail!("prefix step {step_index} after_full_hash mismatch");
            }
        }
    }

    let (computed_prelude, staged) = game.preview_free_three_of_a_kind_if_feasible()?;
    let expected_prelude: MarketPrelude = serde_json::from_value(
        root_replay
            .get("market_prelude")
            .cloned()
            .context("root_replay.market_prelude missing")?,
    )?;
    if computed_prelude != expected_prelude {
        bail!("market prelude mismatch during reconstruction");
    }
    let expected_public = record
        .get("public_hash")
        .and_then(Value::as_str)
        .context("record public_hash missing")?;
    if public_hash(&staged) != expected_public {
        bail!("reconstructed public hash mismatch");
    }
    let expected_full = root_replay
        .get("root_full_hash")
        .and_then(Value::as_str)
        .context("root_replay.root_full_hash missing")?;
    if game_hash(&staged) != expected_full {
        bail!("reconstructed full hash mismatch");
    }

    let legal = staged.legal_turn_actions(&MarketPrelude::default())?;
    let actual_action_ids = legal.iter().map(action_id).collect::<Result<Vec<_>>>()?;
    let expected_action_ids = record
        .get("action_ids")
        .and_then(Value::as_array)
        .context("record action_ids missing")?
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(str::to_owned)
                .context("action_id must be string")
        })
        .collect::<Result<Vec<_>>>()?;
    if actual_action_ids != expected_action_ids {
        bail!("legal action order mismatch");
    }

    let expected_afterstate_hashes = record
        .get("afterstate_hashes")
        .and_then(Value::as_array)
        .context("record afterstate_hashes missing")?
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(str::to_owned)
                .context("afterstate hash must be string")
        })
        .collect::<Result<Vec<_>>>()?;
    for (index, action) in legal.iter().enumerate() {
        let mut after = staged.clone();
        after
            .apply(action)
            .with_context(|| format!("reapplying root action {index}"))?;
        if game_hash(&after) != expected_afterstate_hashes[index] {
            bail!("afterstate hash mismatch for action {index}");
        }
    }

    let selected_action_id = record
        .get("selected_action")
        .and_then(Value::as_str)
        .context("selected_action missing")?;
    let selected_index = actual_action_ids
        .iter()
        .position(|candidate| candidate == selected_action_id)
        .context("selected action is not legal")?;
    let mut selected_after = staged.clone();
    selected_after.apply(&legal[selected_index])?;
    let selected_after_hash = game_hash(&selected_after);
    for sample in record
        .get("chance_samples")
        .and_then(Value::as_array)
        .context("chance_samples missing")?
    {
        if sample.get("action_id").and_then(Value::as_str) == Some(selected_action_id) {
            if sample.get("before_hash").and_then(Value::as_str) != Some(expected_full) {
                bail!("selected chance sample before_hash mismatch");
            }
            if sample.get("after_hash").and_then(Value::as_str)
                != Some(selected_after_hash.as_str())
            {
                bail!("selected chance sample after_hash mismatch");
            }
        }
    }
    Ok(())
}

fn bench_tokenize(args: &Args) -> Result<Value> {
    let started = std::time::Instant::now();
    let mut root_count = 0usize;
    let mut action_count = 0usize;
    let mut token_count = 0usize;
    for offset in 0..args.seed_count {
        let seed_u64 = args.first_seed + offset;
        let config = GameConfig::research_aaaaa(args.player_count)?;
        let mut game = GameState::new(config, GameSeed::from_u64(seed_u64))?;
        for ply_index in 0..args.plies_per_seed {
            if game.is_game_over() {
                break;
            }
            let (prelude, staged) = game.preview_free_three_of_a_kind_if_feasible()?;
            let active_seat = staged.current_player();
            let legal = staged.legal_turn_actions(&MarketPrelude::default())?;
            let public = public_tokens(&staged, active_seat);
            token_count += public
                .get("token_count")
                .and_then(Value::as_u64)
                .unwrap_or(0) as usize;
            for (action_index, action) in legal.iter().enumerate() {
                let mut after = staged.clone();
                after.apply(action)?;
                let score = score_game(&after)[active_seat].base_total;
                let _ = action_token(&staged, &prelude, action, score, active_seat, action_index)?;
            }
            action_count += legal.len();
            root_count += 1;
            let selected = legal
                .first()
                .with_context(|| format!("seed {seed_u64} ply {ply_index} has no legal action"))?
                .clone();
            game = staged;
            game.apply(&selected)?;
        }
    }
    let seconds = started.elapsed().as_secs_f64();
    Ok(json!({
        "status": "pass",
        "mode": "bench_tokenize",
        "seed_count": args.seed_count,
        "plies_per_seed": args.plies_per_seed,
        "roots": root_count,
        "legal_actions": action_count,
        "public_tokens": token_count,
        "elapsed_seconds": seconds,
        "roots_per_second": root_count as f64 / seconds.max(1.0e-9),
        "actions_per_second": action_count as f64 / seconds.max(1.0e-9),
    }))
}

fn validate_hidden_redetermination(args: &Args) -> Result<Value> {
    let mut checked = 0usize;
    for offset in 0..args.seed_count {
        let seed_u64 = args.first_seed + offset;
        let config = GameConfig::research_aaaaa(args.player_count)?;
        let game = GameState::new(config, GameSeed::from_u64(seed_u64))?;
        let public = game.public_state();
        let public_hash_before = public.canonical_hash();
        let supply = game.public_supply();
        let legal = game.legal_turn_actions(&MarketPrelude::default())?;
        let mut redetermined = game.clone();
        redetermined.redeterminize_hidden(GameSeed::from_u64(seed_u64 ^ 0xfeed_face));
        if redetermined.public_state() != public {
            bail!("seed {seed_u64} public state changed after redetermination");
        }
        if redetermined.public_state().canonical_hash() != public_hash_before {
            bail!("seed {seed_u64} public hash changed after redetermination");
        }
        if redetermined.public_supply() != supply {
            bail!("seed {seed_u64} public supply changed after redetermination");
        }
        if redetermined.legal_turn_actions(&MarketPrelude::default())? != legal {
            bail!("seed {seed_u64} legal actions changed after redetermination");
        }
        if redetermined.canonical_hash() == game.canonical_hash() {
            bail!("seed {seed_u64} hidden redetermination did not change hidden hash");
        }
        checked += 1;
    }
    Ok(json!({
        "status": "pass",
        "seeds_checked": checked,
        "first_seed": args.first_seed,
        "seed_count": args.seed_count,
        "public_state_preserved": true,
        "legal_action_order_preserved": true,
        "public_supply_preserved": true,
        "hidden_hash_changes": true,
    }))
}

fn write_expert_search_bench(
    path: &PathBuf,
    records: &[Value],
    elapsed_seconds: f64,
) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let action_count: usize = records
        .iter()
        .map(|record| {
            record
                .get("legal_actions")
                .and_then(Value::as_array)
                .map(Vec::len)
                .unwrap_or(0)
        })
        .sum();
    let report = json!({
        "status": "pass",
        "mode": "chance_mcts_dry_run",
        "roots": records.len(),
        "legal_actions": action_count,
        "elapsed_seconds": elapsed_seconds,
        "roots_per_second": records.len() as f64 / elapsed_seconds.max(1.0e-9),
        "actions_per_second": action_count as f64 / elapsed_seconds.max(1.0e-9),
        "model_fallback": records
            .iter()
            .all(|record| record.pointer("/metadata/model_fallback").and_then(Value::as_bool).unwrap_or(false)),
        "rollouts_per_action": records
            .first()
            .and_then(|record| record.pointer("/metadata/rollouts_per_action"))
            .cloned(),
    });
    std::fs::write(path, format!("{}\n", canonical_json(&report)))?;
    Ok(())
}

fn write_expert_manifest(path: &PathBuf, records: &[Value], args: &Args) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let action_count: usize = records
        .iter()
        .map(|record| {
            record
                .get("legal_actions")
                .and_then(Value::as_array)
                .map(Vec::len)
                .unwrap_or(0)
        })
        .sum();
    let manifest = json!({
        "schema_id": EXPERT_ROOT_SCHEMA_ID,
        "source_generator": "cascadiav3-real-root-exporter",
        "seed_domain": format!(
            "first_seed={},seed_count={},plies_per_seed={},rollouts_per_action={},mode=chance_mcts_dry_run,allow_model_fallback={}",
            args.first_seed,
            args.seed_count,
            args.plies_per_seed,
            args.rollouts_per_action,
            args.allow_model_fallback
        ),
        "record_count": records.len(),
        "checksum": records_checksum(records),
        "scientific_eligibility": "dry_run",
        "created_at_utc": "2026-06-30T00:00:00+00:00",
        "format": "jsonl",
        "ruleset_id": RULESET_ID,
        "legal_action_coverage": 1.0,
        "total_legal_actions": action_count,
        "rollouts_per_action": args.rollouts_per_action,
        "model_fallback": args.allow_model_fallback,
        "rayon_current_num_threads": rayon::current_num_threads(),
        "notes": format!(
            "Expert-root v1 dry-run labels with full legal action coverage and {} sampled greedy continuation(s) per action; not gameplay strength evidence.",
            args.rollouts_per_action
        ),
    });
    std::fs::write(path, format!("{}\n", canonical_json(&manifest)))?;
    Ok(())
}

struct StreamingWriter {
    handle: BufWriter<Box<dyn Write + Send>>,
    digest: Sha256,
    record_count: usize,
}

fn export_greedy_policy_corpus(args: &Args) -> Result<usize> {
    let stdout_mode = args.out.as_os_str() == "-";
    if !stdout_mode {
        if let Some(parent) = args.out.parent() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let handle: Box<dyn Write + Send> = if stdout_mode {
        Box::new(std::io::stdout())
    } else {
        Box::new(File::create(&args.out)?)
    };
    let writer = Arc::new(Mutex::new(StreamingWriter {
        handle: BufWriter::new(handle),
        digest: Sha256::new(),
        record_count: 0,
    }));
    let seed_end = args
        .first_seed
        .checked_add(args.seed_count)
        .context("seed range overflow")?;
    (args.first_seed..seed_end)
        .into_par_iter()
        .try_for_each(|seed_u64| -> Result<()> {
            let records = export_greedy_policy_seed_records(args, seed_u64)?;
            let mut lines = String::new();
            let mut count = 0usize;
            for record in records {
                lines.push_str(&canonical_json(&record));
                lines.push('\n');
                count += 1;
            }
            let mut guard = writer.lock().expect("greedy corpus writer mutex poisoned");
            guard.digest.update(lines.as_bytes());
            guard.record_count += count;
            guard.handle.write_all(lines.as_bytes())?;
            Ok(())
        })?;
    let mut guard = writer.lock().expect("greedy corpus writer mutex poisoned");
    guard.handle.flush()?;
    let record_count = guard.record_count;
    let checksum = format!("{:x}", guard.digest.clone().finalize());
    drop(guard);
    write_stream_manifest(
        &args.manifest,
        args,
        record_count,
        &checksum,
        "behavior_clone_pretraining",
        "Complete greedy self-play roots with selected greedy action labels, no per-action search or rollout teacher.",
    )?;
    Ok(record_count)
}

fn export_greedy_policy_seed_records(args: &Args, seed_u64: u64) -> Result<Vec<Value>> {
    let config = GameConfig::research_aaaaa(args.player_count)?;
    let mut game = GameState::new(config, GameSeed::from_u64(seed_u64))
        .with_context(|| format!("creating greedy policy seed {seed_u64}"))?;
    let mut records = Vec::new();
    let mut ply_index = 0usize;
    while !game.is_game_over() && ply_index < args.plies_per_seed {
        let (root_record, selected_action_id) =
            build_greedy_policy_root_record(&game, seed_u64, ply_index, args)
                .with_context(|| format!("greedy policy seed {seed_u64} ply {ply_index}"))?;
        game = advance_selected_action(&game, args.max_actions, &selected_action_id)?;
        records.push(root_record);
        ply_index += 1;
    }
    let terminal_scores = score_game(&game);
    let score_means = terminal_scores
        .iter()
        .map(|score| ScoreMean {
            wildlife: f64::from(score.wildlife.iter().sum::<u16>()),
            habitat: f64::from(score.habitat.iter().sum::<u16>()),
            nature_tokens: f64::from(score.nature_tokens),
            total: f64::from(score.total),
        })
        .collect::<Vec<_>>();
    let final_score_vector = score_means
        .iter()
        .map(|score| json!(score.total))
        .collect::<Vec<_>>();
    let score_decomposition = score_decomposition(&score_means);
    let rank_vector = rank_vector(&score_means)
        .into_iter()
        .map(|rank| json!(rank))
        .collect::<Vec<_>>();
    for record in &mut records {
        let active_seat = record
            .get("active_seat")
            .and_then(Value::as_u64)
            .unwrap_or(0) as usize;
        if let Some(object) = record.as_object_mut() {
            object.insert(
                "final_score_vector".to_owned(),
                Value::Array(final_score_vector.clone()),
            );
            object.insert(
                "score_decomposition".to_owned(),
                score_decomposition.clone(),
            );
            object.insert("rank_vector".to_owned(), Value::Array(rank_vector.clone()));
            if let Some(metadata) = object.get_mut("metadata").and_then(Value::as_object_mut) {
                metadata.insert(
                    "terminal_active_score".to_owned(),
                    json!(score_means[active_seat].total),
                );
            }
        }
        attach_checksum(record)?;
    }
    Ok(records)
}

fn export_greedy_policy_tensor_corpus(args: &Args) -> Result<usize> {
    if args.out.as_os_str() == "-" {
        bail!("--greedy-policy-tensor-corpus requires a file --out path");
    }
    if let Some(parent) = args.out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let seed_end = args
        .first_seed
        .checked_add(args.seed_count)
        .context("seed range overflow")?;
    let mut per_seed = (args.first_seed..seed_end)
        .into_par_iter()
        .map(|seed_u64| {
            let records = export_greedy_policy_seed_records(args, seed_u64)
                .with_context(|| format!("exporting tensor seed {seed_u64}"))?;
            let shard = TensorShardData::from_records(&records)
                .with_context(|| format!("extracting Rust tensor features for seed {seed_u64}"))?;
            Ok((seed_u64, shard))
        })
        .collect::<Result<Vec<_>>>()?;
    per_seed.sort_by_key(|(seed, _)| *seed);

    let mut shard = TensorShardData::new();
    for (_, seed_shard) in per_seed {
        shard.merge(seed_shard);
    }
    if shard.record_count == 0 {
        bail!("greedy policy tensor exporter produced no records");
    }
    let metadata = tensor_shard_metadata(args, &shard);
    let metadata_json = canonical_json(&metadata);
    npz_writer::write_greedy_tensor_npz(
        &args.out,
        npz_writer::GreedyTensorNpz {
            version: SHARD_VERSION,
            metadata_json: &metadata_json,
            tokens_f16_bits: &shard.tokens_f16_bits,
            token_shape: [shard.total_token_count, PUBLIC_TOKEN_FEATURE_DIM],
            actions_f16_bits: &shard.actions_f16_bits,
            action_shape: [
                shard.total_action_count,
                SEMANTIC_PUBLIC_TOKEN_ACTION_FEATURE_DIM,
            ],
            token_offsets: &shard.token_offsets,
            action_offsets: &shard.action_offsets,
            selected_action_index: &shard.selected_action_index,
            compression: args.tensor_compression,
        },
    )?;
    let checksum = sha256_file_hex(&args.out)?;
    write_tensor_manifest(&args.manifest, args, &shard, &checksum)?;
    Ok(shard.record_count)
}

fn export_expert_tensor_corpus(args: &Args) -> Result<usize> {
    if args.out.as_os_str() == "-" {
        bail!("--expert-tensor-corpus requires a file --out path");
    }
    if let Some(parent) = args.out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let seed_end = args
        .first_seed
        .checked_add(args.seed_count)
        .context("seed range overflow")?;
    let mut per_seed = (args.first_seed..seed_end)
        .into_par_iter()
        .map(|seed_u64| {
            let records = export_expert_seed_records(args, seed_u64)
                .with_context(|| format!("exporting expert tensor seed {seed_u64}"))?;
            let shard = ExpertTensorShardData::from_records(&records).with_context(|| {
                format!("extracting Rust expert tensor features for seed {seed_u64}")
            })?;
            Ok((seed_u64, shard))
        })
        .collect::<Result<Vec<_>>>()?;
    per_seed.sort_by_key(|(seed, _)| *seed);

    let mut shard = ExpertTensorShardData::new();
    for (_, seed_shard) in per_seed {
        shard.merge(seed_shard);
    }
    if shard.record_count == 0 {
        bail!("expert tensor exporter produced no records");
    }
    let metadata = expert_tensor_shard_metadata(args, &shard);
    let metadata_json = canonical_json(&metadata);
    npz_writer::write_expert_tensor_npz(
        &args.out,
        npz_writer::ExpertTensorNpz {
            version: EXPERT_SHARD_VERSION,
            metadata_json: &metadata_json,
            tokens_f16_bits: &shard.tokens_f16_bits,
            token_shape: [shard.total_token_count, PUBLIC_TOKEN_FEATURE_DIM],
            actions_f16_bits: &shard.actions_f16_bits,
            action_shape: [
                shard.total_action_count,
                SEMANTIC_PUBLIC_TOKEN_ACTION_FEATURE_DIM,
            ],
            token_offsets: &shard.token_offsets,
            action_offsets: &shard.action_offsets,
            relation_edges_i32: &shard.relation_edges_i32,
            relation_edge_shape: [shard.total_relation_edge_count, 3],
            relation_offsets: &shard.relation_offsets,
            selected_action_index: &shard.selected_action_index,
            target_q: &shard.target_q,
            target_score_to_go: &shard.target_score_to_go,
            q_valid: &shard.q_valid,
            priors: &shard.priors,
            visits: &shard.visits,
            q_variance: &shard.q_variance,
            q_count: &shard.q_count,
            truncated_count: &shard.truncated_count,
            exact_afterstate_score_active: &shard.exact_afterstate_score_active,
            final_score_vector: &shard.final_score_vector,
            rank_vector: &shard.rank_vector,
            score_decomposition: &shard.score_decomposition,
            improved_policy: None,
            search_root_value: None,
            record_count: shard.record_count,
            compression: args.tensor_compression,
        },
    )?;
    let checksum = sha256_file_hex(&args.out)?;
    write_expert_tensor_manifest(&args.manifest, args, &shard, &checksum)?;
    Ok(shard.record_count)
}

fn export_greedy_expert_tensor_corpus(args: &Args) -> Result<usize> {
    if args.out.as_os_str() == "-" {
        bail!("--greedy-expert-tensor-corpus requires a file --out path");
    }
    if let Some(parent) = args.out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let seed_end = args
        .first_seed
        .checked_add(args.seed_count)
        .context("seed range overflow")?;
    let mut per_seed = (args.first_seed..seed_end)
        .into_par_iter()
        .map(|seed_u64| {
            let records = export_greedy_policy_seed_records(args, seed_u64)
                .with_context(|| format!("exporting greedy expert tensor seed {seed_u64}"))?;
            let shard = ExpertTensorShardData::from_records(&records).with_context(|| {
                format!("extracting Rust greedy expert tensor features for seed {seed_u64}")
            })?;
            Ok((seed_u64, shard))
        })
        .collect::<Result<Vec<_>>>()?;
    per_seed.sort_by_key(|(seed, _)| *seed);

    let mut shard = ExpertTensorShardData::new();
    for (_, seed_shard) in per_seed {
        shard.merge(seed_shard);
    }
    if shard.record_count == 0 {
        bail!("greedy expert tensor exporter produced no records");
    }
    let mut metadata = expert_tensor_shard_metadata(args, &shard);
    if let Some(object) = metadata.as_object_mut() {
        object.insert(
            "source".to_owned(),
            json!("greedy_policy_no_search_expert_tensor_corpus"),
        );
        object.insert(
            "source_paths".to_owned(),
            json!(["rust-native:greedy_expert_tensor_corpus"]),
        );
        object.insert(
            "behavior_policy".to_owned(),
            json!("one_step_greedy_ranker_no_search"),
        );
    }
    let metadata_json = canonical_json(&metadata);
    npz_writer::write_expert_tensor_npz(
        &args.out,
        npz_writer::ExpertTensorNpz {
            version: EXPERT_SHARD_VERSION,
            metadata_json: &metadata_json,
            tokens_f16_bits: &shard.tokens_f16_bits,
            token_shape: [shard.total_token_count, PUBLIC_TOKEN_FEATURE_DIM],
            actions_f16_bits: &shard.actions_f16_bits,
            action_shape: [
                shard.total_action_count,
                SEMANTIC_PUBLIC_TOKEN_ACTION_FEATURE_DIM,
            ],
            token_offsets: &shard.token_offsets,
            action_offsets: &shard.action_offsets,
            relation_edges_i32: &shard.relation_edges_i32,
            relation_edge_shape: [shard.total_relation_edge_count, 3],
            relation_offsets: &shard.relation_offsets,
            selected_action_index: &shard.selected_action_index,
            target_q: &shard.target_q,
            target_score_to_go: &shard.target_score_to_go,
            q_valid: &shard.q_valid,
            priors: &shard.priors,
            visits: &shard.visits,
            q_variance: &shard.q_variance,
            q_count: &shard.q_count,
            truncated_count: &shard.truncated_count,
            exact_afterstate_score_active: &shard.exact_afterstate_score_active,
            final_score_vector: &shard.final_score_vector,
            rank_vector: &shard.rank_vector,
            score_decomposition: &shard.score_decomposition,
            improved_policy: None,
            search_root_value: None,
            record_count: shard.record_count,
            compression: args.tensor_compression,
        },
    )?;
    let checksum = sha256_file_hex(&args.out)?;
    write_expert_tensor_manifest(&args.manifest, args, &shard, &checksum)?;
    Ok(shard.record_count)
}

fn export_greedy_state_search_bootstrap_tensor_corpus(args: &Args) -> Result<usize> {
    if args.out.as_os_str() == "-" {
        bail!("--greedy-state-search-bootstrap-tensor-corpus requires a file --out path");
    }
    if let Some(parent) = args.out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let seed_end = args
        .first_seed
        .checked_add(args.seed_count)
        .context("seed range overflow")?;
    let started = Instant::now();
    let total_seeds = seed_end - args.first_seed;
    let completed_seeds = AtomicU64::new(0);
    let completed_records = AtomicU64::new(0);
    let mut per_seed = (args.first_seed..seed_end)
        .into_par_iter()
        .map(|seed_u64| {
            let records = export_greedy_state_search_bootstrap_seed_records(args, seed_u64)
                .with_context(|| {
                    format!("exporting greedy-state search-bootstrap tensor seed {seed_u64}")
                })?;
            let shard = ExpertTensorShardData::from_records(&records).with_context(|| {
                format!(
                    "extracting Rust greedy-state search-bootstrap tensor features for seed {seed_u64}"
                )
            })?;
            let done = completed_seeds.fetch_add(1, Ordering::Relaxed) + 1;
            let records_done =
                completed_records.fetch_add(shard.record_count as u64, Ordering::Relaxed)
                    + shard.record_count as u64;
            log_seed_export_progress(
                "greedy-state search-bootstrap tensor",
                done,
                total_seeds,
                records_done,
                started,
            );
            Ok((seed_u64, shard))
        })
        .collect::<Result<Vec<_>>>()?;
    per_seed.sort_by_key(|(seed, _)| *seed);

    let mut shard = ExpertTensorShardData::new();
    for (_, seed_shard) in per_seed {
        shard.merge(seed_shard);
    }
    if shard.record_count == 0 {
        bail!("greedy-state search-bootstrap tensor exporter produced no records");
    }
    let mut metadata = expert_tensor_shard_metadata(args, &shard);
    if let Some(object) = metadata.as_object_mut() {
        object.insert(
            "source".to_owned(),
            json!("greedy_state_search_bootstrap_tensor_corpus"),
        );
        object.insert(
            "source_paths".to_owned(),
            json!(["rust-native:greedy_state_search_bootstrap_tensor_corpus"]),
        );
        object.insert(
            "behavior_policy".to_owned(),
            json!("greedy_state_distribution_advance_greedy_action"),
        );
        object.insert(
            "teacher".to_owned(),
            json!("sampled_topk_greedy_rollout_mean_per_retained_action"),
        );
    }
    let metadata_json = canonical_json(&metadata);
    npz_writer::write_expert_tensor_npz(
        &args.out,
        npz_writer::ExpertTensorNpz {
            version: EXPERT_SHARD_VERSION,
            metadata_json: &metadata_json,
            tokens_f16_bits: &shard.tokens_f16_bits,
            token_shape: [shard.total_token_count, PUBLIC_TOKEN_FEATURE_DIM],
            actions_f16_bits: &shard.actions_f16_bits,
            action_shape: [
                shard.total_action_count,
                SEMANTIC_PUBLIC_TOKEN_ACTION_FEATURE_DIM,
            ],
            token_offsets: &shard.token_offsets,
            action_offsets: &shard.action_offsets,
            relation_edges_i32: &shard.relation_edges_i32,
            relation_edge_shape: [shard.total_relation_edge_count, 3],
            relation_offsets: &shard.relation_offsets,
            selected_action_index: &shard.selected_action_index,
            target_q: &shard.target_q,
            target_score_to_go: &shard.target_score_to_go,
            q_valid: &shard.q_valid,
            priors: &shard.priors,
            visits: &shard.visits,
            q_variance: &shard.q_variance,
            q_count: &shard.q_count,
            truncated_count: &shard.truncated_count,
            exact_afterstate_score_active: &shard.exact_afterstate_score_active,
            final_score_vector: &shard.final_score_vector,
            rank_vector: &shard.rank_vector,
            score_decomposition: &shard.score_decomposition,
            improved_policy: None,
            search_root_value: None,
            record_count: shard.record_count,
            compression: args.tensor_compression,
        },
    )?;
    let checksum = sha256_file_hex(&args.out)?;
    write_expert_tensor_manifest(&args.manifest, args, &shard, &checksum)?;
    Ok(shard.record_count)
}

fn export_model_state_search_bootstrap_tensor_corpus(args: &Args) -> Result<usize> {
    if args.out.as_os_str() == "-" {
        bail!("--model-state-search-bootstrap-tensor-corpus requires a file --out path");
    }
    if let Some(parent) = args.out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let seed_end = args
        .first_seed
        .checked_add(args.seed_count)
        .context("seed range overflow")?;
    let started = Instant::now();
    let total_seeds = seed_end - args.first_seed;
    let completed_seeds = AtomicU64::new(0);
    let completed_records = AtomicU64::new(0);
    let seeds = (args.first_seed..seed_end).collect::<Vec<_>>();
    let target_chunks = (rayon::current_num_threads() * 2).max(1);
    let chunk_size = ((seeds.len() + target_chunks - 1) / target_chunks).max(1);
    let per_chunk = seeds
        .par_chunks(chunk_size)
        .map(|seed_chunk| {
            let mut model_session = model_state_worker_session(args)?;
            let mut chunk_shards = Vec::with_capacity(seed_chunk.len());
            for seed_u64 in seed_chunk.iter().copied() {
                let records = export_model_state_search_bootstrap_seed_records_with_session(
                    args,
                    seed_u64,
                    &mut model_session,
                )
                .with_context(|| {
                    format!("exporting model-state search-bootstrap tensor seed {seed_u64}")
                })?;
                let shard = ExpertTensorShardData::from_records(&records).with_context(|| {
                    format!(
                        "extracting Rust model-state search-bootstrap tensor features for seed {seed_u64}"
                    )
                })?;
                let done = completed_seeds.fetch_add(1, Ordering::Relaxed) + 1;
                let records_done =
                    completed_records.fetch_add(shard.record_count as u64, Ordering::Relaxed)
                        + shard.record_count as u64;
                log_seed_export_progress(
                    "model-state search-bootstrap tensor",
                    done,
                    total_seeds,
                    records_done,
                    started,
                );
                chunk_shards.push((seed_u64, shard));
            }
            Ok(chunk_shards)
        })
        .collect::<Result<Vec<_>>>()?;
    let mut per_seed = per_chunk
        .into_iter()
        .flatten()
        .collect::<Vec<(u64, ExpertTensorShardData)>>();
    per_seed.sort_by_key(|(seed, _)| *seed);

    let mut shard = ExpertTensorShardData::new();
    for (_, seed_shard) in per_seed {
        shard.merge(seed_shard);
    }
    if shard.record_count == 0 {
        bail!("model-state search-bootstrap tensor exporter produced no records");
    }
    let mut metadata = expert_tensor_shard_metadata(args, &shard);
    if let Some(object) = metadata.as_object_mut() {
        object.insert(
            "source".to_owned(),
            json!("model_state_search_bootstrap_tensor_corpus"),
        );
        object.insert(
            "source_paths".to_owned(),
            json!(["rust-native:model_state_search_bootstrap_tensor_corpus"]),
        );
        object.insert(
            "behavior_policy".to_owned(),
            json!("model_state_distribution_advance_model_action"),
        );
        object.insert(
            "teacher".to_owned(),
            json!("sampled_topk_greedy_rollout_mean_per_retained_action"),
        );
    }
    let metadata_json = canonical_json(&metadata);
    npz_writer::write_expert_tensor_npz(
        &args.out,
        npz_writer::ExpertTensorNpz {
            version: EXPERT_SHARD_VERSION,
            metadata_json: &metadata_json,
            tokens_f16_bits: &shard.tokens_f16_bits,
            token_shape: [shard.total_token_count, PUBLIC_TOKEN_FEATURE_DIM],
            actions_f16_bits: &shard.actions_f16_bits,
            action_shape: [
                shard.total_action_count,
                SEMANTIC_PUBLIC_TOKEN_ACTION_FEATURE_DIM,
            ],
            token_offsets: &shard.token_offsets,
            action_offsets: &shard.action_offsets,
            relation_edges_i32: &shard.relation_edges_i32,
            relation_edge_shape: [shard.total_relation_edge_count, 3],
            relation_offsets: &shard.relation_offsets,
            selected_action_index: &shard.selected_action_index,
            target_q: &shard.target_q,
            target_score_to_go: &shard.target_score_to_go,
            q_valid: &shard.q_valid,
            priors: &shard.priors,
            visits: &shard.visits,
            q_variance: &shard.q_variance,
            q_count: &shard.q_count,
            truncated_count: &shard.truncated_count,
            exact_afterstate_score_active: &shard.exact_afterstate_score_active,
            final_score_vector: &shard.final_score_vector,
            rank_vector: &shard.rank_vector,
            score_decomposition: &shard.score_decomposition,
            improved_policy: None,
            search_root_value: None,
            record_count: shard.record_count,
            compression: args.tensor_compression,
        },
    )?;
    let checksum = sha256_file_hex(&args.out)?;
    write_expert_tensor_manifest(&args.manifest, args, &shard, &checksum)?;
    Ok(shard.record_count)
}

fn log_seed_export_progress(
    label: &str,
    completed_seeds: u64,
    total_seeds: u64,
    completed_records: u64,
    started: Instant,
) {
    let log_every = if total_seeds <= 10 {
        1
    } else if total_seeds <= 100 {
        10
    } else {
        25
    };
    if completed_seeds != 1 && completed_seeds != total_seeds && completed_seeds % log_every != 0 {
        return;
    }
    let elapsed = started.elapsed().as_secs_f64().max(1.0e-9);
    let seed_rate = completed_seeds as f64 / elapsed;
    let record_rate = completed_records as f64 / elapsed;
    eprintln!(
        "[real-root-exporter] {label}: completed {completed_seeds}/{total_seeds} seeds, {completed_records} records, elapsed {:.1}s, {:.3} seeds/s, {:.3} records/s",
        elapsed, seed_rate, record_rate
    );
}

fn export_greedy_state_search_bootstrap_seed_records(
    args: &Args,
    seed_u64: u64,
) -> Result<Vec<Value>> {
    let config = GameConfig::research_aaaaa(args.player_count)?;
    let mut game = GameState::new(config, GameSeed::from_u64(seed_u64))
        .with_context(|| format!("creating greedy-state search-bootstrap seed {seed_u64}"))?;
    let mut records = Vec::new();
    let mut ply_index = 0usize;
    while !game.is_game_over() && ply_index < args.plies_per_seed {
        let mut root_record =
            build_root_record(&game, seed_u64, ply_index, args).with_context(|| {
                format!("greedy-state search-bootstrap seed {seed_u64} ply {ply_index}")
            })?;
        let greedy_action_id = root_record
            .get("legal_actions")
            .and_then(Value::as_array)
            .and_then(|actions| actions.first())
            .and_then(|action| action.get("action_id"))
            .and_then(Value::as_str)
            .context("greedy-state search-bootstrap root missing greedy legal action id")?
            .to_owned();
        let selected_action_id = root_record
            .get("selected_action")
            .and_then(Value::as_str)
            .context("greedy-state search-bootstrap root missing selected action")?
            .to_owned();
        let teacher_advantage = root_record
            .get("per_action_Q")
            .and_then(Value::as_array)
            .and_then(|q_values| {
                let selected_index = root_record
                    .get("legal_actions")
                    .and_then(Value::as_array)?
                    .iter()
                    .position(|action| {
                        action.get("action_id").and_then(Value::as_str)
                            == Some(selected_action_id.as_str())
                    })?;
                Some(
                    q_values[selected_index].as_f64().unwrap_or(0.0)
                        - q_values.first().and_then(Value::as_f64).unwrap_or(0.0),
                )
            })
            .unwrap_or(0.0);
        if let Some(metadata) = root_record
            .as_object_mut()
            .and_then(|object| object.get_mut("metadata"))
            .and_then(Value::as_object_mut)
        {
            metadata.insert(
                "source".to_owned(),
                json!("greedy_state_search_bootstrap_tensor_corpus"),
            );
            metadata.insert(
                "scientific_eligibility".to_owned(),
                json!("expert_iteration_bootstrap"),
            );
            metadata.insert(
                "behavior_policy".to_owned(),
                json!("greedy_state_distribution_advance_greedy_action"),
            );
            metadata.insert(
                "greedy_action_id".to_owned(),
                json!(greedy_action_id.clone()),
            );
            metadata.insert(
                "selected_is_greedy".to_owned(),
                json!(selected_action_id == greedy_action_id),
            );
            metadata.insert(
                "teacher_advantage_over_greedy".to_owned(),
                json!(teacher_advantage),
            );
        }
        attach_checksum(&mut root_record)?;
        game = advance_selected_action(&game, args.max_actions, &greedy_action_id)?;
        records.push(root_record);
        ply_index += 1;
    }
    Ok(records)
}

fn model_state_worker_session(args: &Args) -> Result<Option<ModelServiceSession>> {
    match args.model_service.as_ref() {
        Some(command) => match ModelServiceSession::spawn(command, &BridgeConfig::from_args(args)) {
            Ok(session) => Ok(Some(session)),
            Err(error) if args.allow_model_fallback => {
                eprintln!("model service unavailable for worker; using fallback priors: {error}");
                Ok(None)
            }
            Err(error) => Err(error),
        },
        None => Ok(None),
    }
}

fn export_model_state_search_bootstrap_seed_records_with_session(
    args: &Args,
    seed_u64: u64,
    model_session: &mut Option<ModelServiceSession>,
) -> Result<Vec<Value>> {
    let config = GameConfig::research_aaaaa(args.player_count)?;
    let mut game = GameState::new(config, GameSeed::from_u64(seed_u64))
        .with_context(|| format!("creating model-state search-bootstrap seed {seed_u64}"))?;
    let mut records = Vec::new();
    let mut ply_index = 0usize;
    while !game.is_game_over() && ply_index < args.plies_per_seed {
        let mut root_record =
            build_root_record(&game, seed_u64, ply_index, args).with_context(|| {
                format!("model-state search-bootstrap seed {seed_u64} ply {ply_index}")
            })?;
        let (model_request, action_ids) = model_request_for_tensor_root(&root_record)?;
        let model_eval = model_eval_for_root_with_optional_session(
            args,
            &model_request,
            action_ids.len(),
            model_session,
        )
        .with_context(|| format!("model-state eval seed {seed_u64} ply {ply_index}"))?;
        let (model_action_id, model_selection_head, model_selection_score) =
            model_selected_action(&action_ids, &model_eval)?;
        let selected_action_id = root_record
            .get("selected_action")
            .and_then(Value::as_str)
            .context("model-state search-bootstrap root missing selected action")?
            .to_owned();
        let model_action_index = action_ids
            .iter()
            .position(|action_id| action_id == &model_action_id)
            .context("model selected action id not found in root action_ids")?;
        let selected_action_index = action_ids
            .iter()
            .position(|action_id| action_id == &selected_action_id)
            .context("teacher selected action id not found in root action_ids")?;
        let teacher_advantage_over_model = root_record
            .get("per_action_Q")
            .and_then(Value::as_array)
            .map(|q_values| {
                q_values[selected_action_index].as_f64().unwrap_or(0.0)
                    - q_values[model_action_index].as_f64().unwrap_or(0.0)
            })
            .unwrap_or(0.0);
        if let Some(metadata) = root_record
            .as_object_mut()
            .and_then(|object| object.get_mut("metadata"))
            .and_then(Value::as_object_mut)
        {
            metadata.insert(
                "source".to_owned(),
                json!("model_state_search_bootstrap_tensor_corpus"),
            );
            metadata.insert(
                "scientific_eligibility".to_owned(),
                json!("expert_iteration_model_state_bootstrap"),
            );
            metadata.insert(
                "behavior_policy".to_owned(),
                json!("model_state_distribution_advance_model_action"),
            );
            metadata.insert("model_action_id".to_owned(), json!(model_action_id.clone()));
            metadata.insert(
                "model_selection_head".to_owned(),
                json!(model_selection_head),
            );
            metadata.insert(
                "model_selection_score".to_owned(),
                json!(model_selection_score),
            );
            metadata.insert(
                "selected_is_model_action".to_owned(),
                json!(selected_action_id == model_action_id),
            );
            metadata.insert(
                "teacher_advantage_over_model_action".to_owned(),
                json!(teacher_advantage_over_model),
            );
            metadata.insert(
                "model_fallback".to_owned(),
                json!(model_eval.model_fallback),
            );
            metadata.insert(
                "model_q_available".to_owned(),
                json!(model_eval.q.is_some()),
            );
            metadata.insert(
                "model_score_to_go_available".to_owned(),
                json!(model_eval.score_to_go.is_some()),
            );
        }
        attach_checksum(&mut root_record)?;
        game = advance_selected_action(&game, args.max_actions, &model_action_id)?;
        records.push(root_record);
        ply_index += 1;
    }
    Ok(records)
}

fn model_eval_for_root_with_optional_session(
    args: &Args,
    root_request: &Value,
    action_count: usize,
    session: &mut Option<ModelServiceSession>,
) -> Result<ModelEval> {
    if let Some(client) = session.as_mut() {
        match client.eval(root_request) {
            Ok(eval) => return Ok(eval),
            Err(error) if args.allow_model_fallback => {
                eprintln!("model service eval failed; falling back to uniform priors: {error}");
                *session = None;
                return Ok(uniform_model_eval(action_count));
            }
            Err(error) => return Err(error),
        }
    }
    if args.model_service.is_some() && args.allow_model_fallback {
        return Ok(uniform_model_eval(action_count));
    }
    model_eval_for_root(args, root_request, action_count)
}

fn model_request_for_tensor_root(root_record: &Value) -> Result<(Value, Vec<String>)> {
    let legal_actions = root_record
        .get("legal_actions")
        .and_then(Value::as_array)
        .context("tensor root legal_actions missing")?;
    let mut action_ids = Vec::with_capacity(legal_actions.len());
    for (index, action) in legal_actions.iter().enumerate() {
        let action_id = action
            .get("action_id")
            .and_then(Value::as_str)
            .with_context(|| format!("tensor root legal action {index} missing action_id"))?;
        action_ids.push(action_id.to_owned());
    }
    let request = json!({
        "schema_id": root_record.get("schema_id").cloned().unwrap_or_else(|| json!(EXPERT_ROOT_SCHEMA_ID)),
        "state_hash": root_record.get("state_hash").cloned().context("tensor root state_hash missing")?,
        "active_seat": root_record.get("active_seat").cloned().context("tensor root active_seat missing")?,
        "legal_actions": root_record.get("legal_actions").cloned().context("tensor root legal_actions missing")?,
        "action_ids": action_ids.iter().map(|action_id| json!(action_id)).collect::<Vec<_>>(),
        "exact_afterstate_score_active": root_record
            .get("exact_afterstate_score_active")
            .cloned()
            .context("tensor root exact_afterstate_score_active missing")?,
        "public_tokens": root_record.get("public_tokens").cloned().context("tensor root public_tokens missing")?,
    });
    Ok((request, action_ids))
}

fn model_selected_action(
    action_ids: &[String],
    model_eval: &ModelEval,
) -> Result<(String, String, f64)> {
    if action_ids.is_empty() {
        bail!("cannot select a model action from an empty action menu");
    }
    if let Some(q_values) = &model_eval.q {
        let (index, value) = q_values
            .iter()
            .enumerate()
            .max_by(|(left_index, left), (right_index, right)| {
                left.partial_cmp(right)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| right_index.cmp(left_index))
            })
            .context("model q vector was empty")?;
        return Ok((action_ids[index].clone(), "q".to_owned(), *value));
    }
    let mut best_index = 0usize;
    let mut best_prior = f64::NEG_INFINITY;
    for (index, value) in model_eval.priors.iter().enumerate() {
        let prior = value.as_f64().context("model prior was not numeric")?;
        if prior > best_prior {
            best_prior = prior;
            best_index = index;
        }
    }
    Ok((
        action_ids[best_index].clone(),
        "prior".to_owned(),
        best_prior,
    ))
}

fn gumbel_root_menu_limit(args: &Args) -> Option<usize> {
    if args.gumbel_root_menu == 0 {
        None
    } else {
        Some(args.gumbel_root_menu)
    }
}

fn gumbel_config_from_args(args: &Args, search_seed: u64) -> gumbel::GumbelConfig {
    gumbel::GumbelConfig {
        n_simulations: args.gumbel_n_simulations,
        top_m: args.gumbel_top_m,
        max_root_actions: args.gumbel_max_root_actions,
        depth_rounds: args.gumbel_depth_rounds,
        determinization_samples: args.gumbel_determinizations,
        rollout_blend_weight: args.gumbel_blend_weight,
        rollout_max_actions: args.max_actions,
        rollout_top_k: args.rollout_top_k,
        k_interior: args.k_interior,
        exploration: args.gumbel_exploration,
        search_seed,
        ..gumbel::GumbelConfig::default()
    }
}

fn gumbel_search_seed(seed_u64: u64, ply_index: usize) -> u64 {
    gumbel::splitmix64(seed_u64 ^ gumbel::splitmix64(ply_index as u64 ^ 0x6706_2026))
}

fn eval_request_for_row(row: &gumbel::EvalRow) -> Result<Value> {
    let staged = &row.staged;
    let active_seat = staged.current_player();
    let mut legal_actions = Vec::with_capacity(row.afterstates.len());
    let mut action_ids = Vec::with_capacity(row.afterstates.len());
    for (index, afterstate) in row.afterstates.iter().enumerate() {
        legal_actions.push(action_token(
            staged,
            &row.prelude,
            &afterstate.candidate.action,
            afterstate.candidate.resulting_base_score,
            active_seat,
            index,
        )?);
        action_ids.push(json!(action_id(&afterstate.candidate.action)?));
    }
    let public_hash = staged.public_state().canonical_hash();
    Ok(json!({
        "schema_id": EXPERT_ROOT_SCHEMA_ID,
        "state_hash": format!("blake3:{}", public_hash.to_hex()),
        "active_seat": active_seat,
        "legal_actions": legal_actions,
        "action_ids": action_ids,
        "exact_afterstate_score_active": row
            .afterstates
            .iter()
            .map(|afterstate| json!(afterstate.exact_score_active))
            .collect::<Vec<_>>(),
        "public_tokens": public_tokens(staged, active_seat),
    }))
}

/// Per-worker bridge handle: either an owned session (one CUDA context per
/// worker) or a client of the process-wide shared aggregator bridge (one
/// CUDA context total, cross-worker batching).
enum ChunkBridge {
    Owned(Option<ModelServiceSession>),
    Shared(SharedBridgeClient),
}

/// Batched leaf evaluator over the JSONL model bridge. Falls back to
/// exact-afterstate-only values (uniform priors) when the bridge is
/// unavailable and fallback is allowed.
struct BridgeLeafEvaluator<'a> {
    bridge: &'a mut ChunkBridge,
    allow_model_fallback: bool,
}

impl gumbel::LeafEvaluator for BridgeLeafEvaluator<'_> {
    fn evaluate_batch(&mut self, rows: &[gumbel::EvalRow]) -> Result<Vec<gumbel::EvalOut>> {
        let requests = rows
            .iter()
            .map(eval_request_for_row)
            .collect::<Result<Vec<_>>>()?;
        let evals: Vec<ModelEval> = match self.bridge {
            ChunkBridge::Shared(client) => client.eval_batch(&requests)?,
            ChunkBridge::Owned(session_slot) => {
                if let Some(session) = session_slot.as_mut() {
                    match session.eval_batch(&requests) {
                        Ok(evals) => evals,
                        Err(error) if self.allow_model_fallback => {
                            eprintln!(
                                "model service batch eval failed; falling back to uniform priors: {error}"
                            );
                            *session_slot = None;
                            rows.iter()
                                .map(|row| uniform_model_eval(row.afterstates.len()))
                                .collect()
                        }
                        Err(error) => return Err(error),
                    }
                } else if self.allow_model_fallback {
                    rows.iter()
                        .map(|row| uniform_model_eval(row.afterstates.len()))
                        .collect()
                } else {
                    bail!("gumbel search requires a model service or --allow-model-fallback");
                }
            }
        };
        rows.iter()
            .zip(evals.into_iter())
            .map(|(row, eval)| {
                let action_count = row.afterstates.len();
                if eval.priors.len() != action_count {
                    bail!("model priors misaligned with gumbel row menu");
                }
                let priors = eval
                    .priors
                    .iter()
                    .map(|value| value.as_f64().context("model prior must be numeric"))
                    .collect::<Result<Vec<_>>>()?;
                // Prefer exact + score_to_go: identical to the bridge's own q
                // for real models, and degrades to exact-afterstate values
                // (score_to_go == 0) under uniform fallback.
                let derived_final_q = if let Some(score_to_go) = &eval.score_to_go {
                    if score_to_go.len() != action_count {
                        bail!("model score_to_go misaligned with gumbel row menu");
                    }
                    row.afterstates
                        .iter()
                        .zip(score_to_go.iter())
                        .map(|(afterstate, remaining)| afterstate.exact_score_active + remaining)
                        .collect()
                } else if let Some(q_values) = &eval.q {
                    if q_values.len() != action_count {
                        bail!("model q misaligned with gumbel row menu");
                    }
                    q_values.clone()
                } else {
                    row.afterstates
                        .iter()
                        .map(|afterstate| afterstate.exact_score_active)
                        .collect()
                };
                Ok(gumbel::EvalOut {
                    priors,
                    derived_final_q,
                })
            })
            .collect()
    }
}

fn score_means_from_breakdowns(scores: &[ScoreBreakdown]) -> Vec<ScoreMean> {
    scores
        .iter()
        .map(|score| ScoreMean {
            wildlife: f64::from(score.wildlife.iter().sum::<u16>()),
            habitat: f64::from(score.habitat.iter().sum::<u16>()),
            nature_tokens: f64::from(score.nature_tokens),
            total: f64::from(score.total),
        })
        .collect()
}

fn gumbel_search_metadata(args: &Args, result: &gumbel::GumbelSearchResult) -> Value {
    json!({
        "n_simulations": args.gumbel_n_simulations,
        "top_m": args.gumbel_top_m,
        "depth_rounds": args.gumbel_depth_rounds,
        "determinization_samples": args.gumbel_determinizations,
        "rollout_blend_weight": args.gumbel_blend_weight,
        "exploration": args.gumbel_exploration,
        "k_interior": args.k_interior,
        "max_root_actions": args.gumbel_max_root_actions,
        "root_menu": args.gumbel_root_menu,
        "simulations_run": result.simulations_run,
    })
}

fn gumbel_selfplay_root_record(
    row: &gumbel::EvalRow,
    result: &gumbel::GumbelSearchResult,
    seed_u64: u64,
    ply_index: usize,
    args: &Args,
) -> Result<Value> {
    let staged = &row.staged;
    let active_seat = staged.current_player();
    let action_count = row.afterstates.len();
    let mut legal_actions = Vec::with_capacity(action_count);
    for (index, afterstate) in row.afterstates.iter().enumerate() {
        legal_actions.push(action_token(
            staged,
            &row.prelude,
            &afterstate.candidate.action,
            afterstate.candidate.resulting_base_score,
            active_seat,
            index,
        )?);
    }
    let selected_action_id = action_id(&row.afterstates[result.chosen_index].candidate.action)?;
    let exact_scores: Vec<f64> = row
        .afterstates
        .iter()
        .map(|afterstate| afterstate.exact_score_active)
        .collect();
    let public_hash = staged.public_state().canonical_hash();
    let placeholder_means = vec![
        ScoreMean {
            wildlife: 0.0,
            habitat: 0.0,
            nature_tokens: 0.0,
            total: 0.0,
        };
        4
    ];
    Ok(json!({
        "schema_id": SCHEMA_ID,
        "state_hash": format!("blake3:{}", public_hash.to_hex()),
        "active_seat": active_seat,
        "legal_actions": legal_actions,
        "priors": result.root_priors,
        "visits": result.visit_counts,
        "per_action_Q": result.completed_q,
        "per_action_score_to_go": result
            .completed_q
            .iter()
            .zip(exact_scores.iter())
            .map(|(q_value, exact)| json!(q_value - exact))
            .collect::<Vec<_>>(),
        "per_action_Q_variance": result.value_variance,
        "per_action_Q_count": result
            .visit_counts
            .iter()
            .map(|visits| json!(visits))
            .collect::<Vec<_>>(),
        "per_action_truncated_count": vec![json!(0); action_count],
        "exact_afterstate_score_active": exact_scores,
        "per_action_Q_valid": result
            .visit_counts
            .iter()
            .map(|visits| json!(*visits > 0))
            .collect::<Vec<_>>(),
        "selected_action": selected_action_id,
        "improved_policy": result.improved_policy,
        "search_root_value": result.root_value,
        "chance_samples": [],
        // Outcome labels are placeholders until the game finishes; see
        // backfill_final_outcome.
        "final_score_vector": vec![json!(0.0); 4],
        "score_decomposition": score_decomposition(&placeholder_means),
        "rank_vector": vec![json!(0); 4],
        "public_tokens": public_tokens(staged, active_seat),
        "metadata": {
            "source": "gumbel_selfplay_tensor_corpus",
            "scientific_eligibility": "gumbel_selfplay_expert_iteration",
            "behavior_policy": "gumbel_search_all_seats_advance_chosen",
            "teacher": "gumbel_completed_q_with_real_outcome_values",
            "root_seed_u64": seed_u64,
            "ply_index_for_seed": ply_index,
            "completed_turns": staged.completed_turns(),
            "turns_remaining": staged.turns_remaining(),
            "retained_action_count": action_count,
            "max_actions": args.max_actions,
            "search": gumbel_search_metadata(args, result),
        },
    }))
}

fn backfill_final_outcome(records: &mut [Value], final_scores: &[ScoreBreakdown]) -> Result<()> {
    let means = score_means_from_breakdowns(final_scores);
    let final_score_vector: Vec<Value> = means.iter().map(|mean| json!(mean.total)).collect();
    let decomposition = score_decomposition(&means);
    let ranks: Vec<Value> = rank_vector(&means).into_iter().map(|rank| json!(rank)).collect();
    for record in records.iter_mut() {
        let object = record
            .as_object_mut()
            .context("selfplay record must be an object")?;
        object.insert("final_score_vector".to_owned(), json!(final_score_vector));
        object.insert("score_decomposition".to_owned(), decomposition.clone());
        object.insert("rank_vector".to_owned(), json!(ranks));
        attach_checksum(record)?;
    }
    Ok(())
}

fn play_gumbel_selfplay_seed(
    args: &Args,
    seed_u64: u64,
    bridge: &mut ChunkBridge,
) -> Result<Vec<Value>> {
    let config = GameConfig::research_aaaaa(args.player_count)?;
    let mut game = GameState::new(config, GameSeed::from_u64(seed_u64))
        .with_context(|| format!("creating gumbel selfplay seed {seed_u64}"))?;
    let mut records = Vec::new();
    let mut ply_index = 0usize;
    while !game.is_game_over() && ply_index < args.plies_per_seed {
        let Some(row) = gumbel::eval_row_for_state(&game, gumbel_root_menu_limit(args))? else {
            break;
        };
        let cfg = gumbel_config_from_args(args, gumbel_search_seed(seed_u64, ply_index));
        let mut evaluator = BridgeLeafEvaluator {
            bridge,
            allow_model_fallback: args.allow_model_fallback,
        };
        let result = gumbel::gumbel_search(&row, &mut evaluator, &cfg)
            .with_context(|| format!("gumbel selfplay seed {seed_u64} ply {ply_index}"))?;
        let record = gumbel_selfplay_root_record(&row, &result, seed_u64, ply_index, args)?;
        records.push(record);
        let chosen = &row.afterstates[result.chosen_index];
        game = chosen.state.clone();
        if chosen.apply_truncated {
            break;
        }
        ply_index += 1;
    }
    let final_scores = score_game(&game);
    backfill_final_outcome(&mut records, &final_scores)?;
    Ok(records)
}

fn export_gumbel_selfplay_tensor_corpus(args: &Args) -> Result<usize> {
    if args.out.as_os_str() == "-" {
        bail!("--gumbel-selfplay-tensor-corpus requires a file --out path");
    }
    if let Some(parent) = args.out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let seed_end = args
        .first_seed
        .checked_add(args.seed_count)
        .context("seed range overflow")?;
    let started = Instant::now();
    let total_seeds = seed_end - args.first_seed;
    let completed_seeds = AtomicU64::new(0);
    let completed_records = AtomicU64::new(0);
    let seeds = (args.first_seed..seed_end).collect::<Vec<_>>();
    let target_chunks = args
        .model_sessions
        .unwrap_or_else(|| (rayon::current_num_threads() * 2).max(1))
        .max(1);
    let chunk_size = ((seeds.len() + target_chunks - 1) / target_chunks).max(1);
    // Shared bridge: one CUDA context serving all chunks with cross-chunk
    // request batching. `--model-sessions` then only controls how many games
    // run in parallel.
    let shared_bridge = if args.shared_model_session {
        let command = args
            .model_service
            .as_ref()
            .context("--shared-model-session requires --model-service")?;
        Some(SharedBridge::spawn(
            command,
            &BridgeConfig::from_args(args),
            192,
        )?)
    } else {
        None
    };
    let per_chunk = seeds
        .par_chunks(chunk_size)
        .map(|seed_chunk| {
            let mut chunk_bridge = match &shared_bridge {
                Some(shared) => ChunkBridge::Shared(shared.client()),
                None => ChunkBridge::Owned(model_state_worker_session(args)?),
            };
            let mut chunk_shards = Vec::with_capacity(seed_chunk.len());
            for seed_u64 in seed_chunk.iter().copied() {
                let records = play_gumbel_selfplay_seed(args, seed_u64, &mut chunk_bridge)
                    .with_context(|| format!("exporting gumbel selfplay seed {seed_u64}"))?;
                let shard = ExpertTensorShardData::from_records(&records).with_context(|| {
                    format!("extracting gumbel selfplay tensor features for seed {seed_u64}")
                })?;
                let done = completed_seeds.fetch_add(1, Ordering::Relaxed) + 1;
                let records_done =
                    completed_records.fetch_add(shard.record_count as u64, Ordering::Relaxed)
                        + shard.record_count as u64;
                log_seed_export_progress(
                    "gumbel selfplay tensor",
                    done,
                    total_seeds,
                    records_done,
                    started,
                );
                chunk_shards.push((seed_u64, shard));
            }
            Ok(chunk_shards)
        })
        .collect::<Result<Vec<_>>>()?;
    let mut per_seed = per_chunk
        .into_iter()
        .flatten()
        .collect::<Vec<(u64, ExpertTensorShardData)>>();
    per_seed.sort_by_key(|(seed, _)| *seed);

    let mut shard = ExpertTensorShardData::new();
    for (_, seed_shard) in per_seed {
        shard.merge(seed_shard);
    }
    if shard.record_count == 0 {
        bail!("gumbel selfplay tensor exporter produced no records");
    }
    if shard.improved_policy_records != shard.record_count {
        bail!(
            "gumbel selfplay shard has {} improved-policy records for {} records",
            shard.improved_policy_records,
            shard.record_count
        );
    }
    let mut metadata = expert_tensor_shard_metadata(args, &shard);
    if let Some(object) = metadata.as_object_mut() {
        object.insert("version".to_owned(), json!(EXPERT_SHARD_VERSION_V2));
        object.insert("schema_id".to_owned(), json!(EXPERT_TENSOR_SCHEMA_ID_V2));
        object.insert("source".to_owned(), json!("gumbel_selfplay_tensor_corpus"));
        object.insert(
            "source_paths".to_owned(),
            json!(["rust-native:gumbel_selfplay_tensor_corpus"]),
        );
        object.insert(
            "behavior_policy".to_owned(),
            json!("gumbel_search_all_seats_advance_chosen"),
        );
        object.insert(
            "teacher".to_owned(),
            json!("gumbel_completed_q_with_real_outcome_values"),
        );
        object.insert(
            "search".to_owned(),
            json!({
                "n_simulations": args.gumbel_n_simulations,
                "top_m": args.gumbel_top_m,
                "depth_rounds": args.gumbel_depth_rounds,
                "determinization_samples": args.gumbel_determinizations,
                "rollout_blend_weight": args.gumbel_blend_weight,
                "exploration": args.gumbel_exploration,
                "k_interior": args.k_interior,
                "max_root_actions": args.gumbel_max_root_actions,
                "root_menu": args.gumbel_root_menu,
            }),
        );
    }
    let metadata_json = canonical_json(&metadata);
    npz_writer::write_expert_tensor_npz(
        &args.out,
        npz_writer::ExpertTensorNpz {
            version: feature_tensors::EXPERT_SHARD_VERSION_V2,
            metadata_json: &metadata_json,
            tokens_f16_bits: &shard.tokens_f16_bits,
            token_shape: [shard.total_token_count, PUBLIC_TOKEN_FEATURE_DIM],
            actions_f16_bits: &shard.actions_f16_bits,
            action_shape: [
                shard.total_action_count,
                SEMANTIC_PUBLIC_TOKEN_ACTION_FEATURE_DIM,
            ],
            token_offsets: &shard.token_offsets,
            action_offsets: &shard.action_offsets,
            relation_edges_i32: &shard.relation_edges_i32,
            relation_edge_shape: [shard.total_relation_edge_count, 3],
            relation_offsets: &shard.relation_offsets,
            selected_action_index: &shard.selected_action_index,
            target_q: &shard.target_q,
            target_score_to_go: &shard.target_score_to_go,
            q_valid: &shard.q_valid,
            priors: &shard.priors,
            visits: &shard.visits,
            q_variance: &shard.q_variance,
            q_count: &shard.q_count,
            truncated_count: &shard.truncated_count,
            exact_afterstate_score_active: &shard.exact_afterstate_score_active,
            final_score_vector: &shard.final_score_vector,
            rank_vector: &shard.rank_vector,
            score_decomposition: &shard.score_decomposition,
            improved_policy: Some(&shard.improved_policy),
            search_root_value: Some(&shard.search_root_value),
            record_count: shard.record_count,
            compression: args.tensor_compression,
        },
    )?;
    let checksum = sha256_file_hex(&args.out)?;
    write_expert_tensor_manifest(&args.manifest, args, &shard, &checksum)?;
    Ok(shard.record_count)
}

fn run_gumbel_policy_game(args: &Args) -> Result<()> {
    let seed_end = args
        .first_seed
        .checked_add(args.seed_count)
        .context("seed range overflow")?;
    let mut chunk_bridge = ChunkBridge::Owned(model_state_worker_session(args)?);
    let mut records = Vec::new();
    for seed_u64 in args.first_seed..seed_end {
        let config = GameConfig::research_aaaaa(args.player_count)?;
        let mut game = GameState::new(config, GameSeed::from_u64(seed_u64))
            .with_context(|| format!("creating gumbel policy game seed {seed_u64}"))?;
        let game_started = Instant::now();
        let mut ply_index = 0usize;
        while !game.is_game_over() {
            let Some(row) = gumbel::eval_row_for_state(&game, gumbel_root_menu_limit(args))? else {
                break;
            };
            let decision_started = Instant::now();
            let cfg = gumbel_config_from_args(args, gumbel_search_seed(seed_u64, ply_index));
            let mut evaluator = BridgeLeafEvaluator {
                bridge: &mut chunk_bridge,
                allow_model_fallback: args.allow_model_fallback,
            };
            let result = gumbel::gumbel_search(&row, &mut evaluator, &cfg)
                .with_context(|| format!("gumbel policy game seed {seed_u64} ply {ply_index}"))?;
            let chosen = &row.afterstates[result.chosen_index];
            records.push(json!({
                "type": "gumbel_decision",
                "seed": seed_u64,
                "ply": ply_index,
                "active_seat": row.staged.current_player(),
                "action_count": row.afterstates.len(),
                "chosen_action_id": action_id(&chosen.candidate.action)?,
                "root_value": result.root_value,
                "simulations_run": result.simulations_run,
                "decision_seconds": decision_started.elapsed().as_secs_f64(),
            }));
            game = chosen.state.clone();
            if chosen.apply_truncated {
                break;
            }
            ply_index += 1;
            if ply_index > 120 {
                bail!("gumbel policy game exceeded expected turn guard");
            }
        }
        let scores = score_game(&game);
        records.push(json!({
            "type": "gumbel_game_done",
            "seed": seed_u64,
            "scores": scores.iter().map(score_breakdown_json).collect::<Vec<_>>(),
            "decision_count": ply_index,
            "elapsed_seconds": game_started.elapsed().as_secs_f64(),
            "search": {
                "n_simulations": args.gumbel_n_simulations,
                "top_m": args.gumbel_top_m,
                "depth_rounds": args.gumbel_depth_rounds,
                "determinization_samples": args.gumbel_determinizations,
                "rollout_blend_weight": args.gumbel_blend_weight,
                "exploration": args.gumbel_exploration,
                "k_interior": args.k_interior,
            },
        }));
        eprintln!(
            "gumbel policy game seed {seed_u64} complete: {} decisions in {:.1}s",
            ply_index,
            game_started.elapsed().as_secs_f64()
        );
    }
    write_jsonl(&args.out, &records)?;
    Ok(())
}

fn build_greedy_policy_root_record(
    game: &GameState,
    seed_u64: u64,
    ply_index: usize,
    args: &Args,
) -> Result<(Value, String)> {
    let active_seat = game.current_player();
    let (prelude, staged) = game.preview_free_three_of_a_kind_if_feasible()?;
    let candidates =
        rank_greedy_actions(&staged, &MarketPrelude::default(), Some(args.max_actions))?;
    if candidates.is_empty() {
        bail!("no legal greedy candidates for policy corpus root");
    }
    let exact_afterstate_scores =
        exact_afterstate_scores_for_candidates(&staged, &candidates, active_seat)?;
    let legal_actions = candidates
        .iter()
        .enumerate()
        .map(|(index, candidate)| {
            action_token(
                &staged,
                &prelude,
                &candidate.action,
                candidate.resulting_base_score,
                active_seat,
                index,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let action_count = legal_actions.len();
    let selected_action_id = action_id(&candidates[0].action)?;
    let public_hash = staged.public_state().canonical_hash();
    let priors = (0..action_count)
        .map(|index| json!(if index == 0 { 1.0 } else { 0.0 }))
        .collect::<Vec<_>>();
    let visits = (0..action_count)
        .map(|index| json!(if index == 0 { 1 } else { 0 }))
        .collect::<Vec<_>>();
    let per_action_q = (0..action_count)
        .map(|index| json!(if index == 0 { 1.0 } else { 0.0 }))
        .collect::<Vec<_>>();
    let exact_afterstate_score_active = exact_afterstate_scores
        .iter()
        .map(|score| json!(score))
        .collect::<Vec<_>>();
    let record = json!({
        "schema_id": SCHEMA_ID,
        "state_hash": format!("blake3:{}", public_hash.to_hex()),
        "active_seat": active_seat,
        "legal_actions": legal_actions,
        "priors": priors,
        "visits": visits,
        "per_action_Q": per_action_q.clone(),
        "per_action_score_to_go": per_action_q,
        "per_action_Q_valid": vec![json!(true); action_count],
        "per_action_Q_variance": vec![json!(0.0); action_count],
        "per_action_Q_count": vec![json!(1.0); action_count],
        "per_action_truncated_count": vec![json!(0); action_count],
        "exact_afterstate_score_active": exact_afterstate_score_active,
        "selected_action": selected_action_id,
        "chance_samples": [],
        "final_score_vector": vec![json!(0.0); 4],
        "score_decomposition": {
            "0": {"wildlife": 0.0, "habitat": 0.0, "nature_tokens": 0.0, "total": 0.0},
            "1": {"wildlife": 0.0, "habitat": 0.0, "nature_tokens": 0.0, "total": 0.0},
            "2": {"wildlife": 0.0, "habitat": 0.0, "nature_tokens": 0.0, "total": 0.0},
            "3": {"wildlife": 0.0, "habitat": 0.0, "nature_tokens": 0.0, "total": 0.0}
        },
        "rank_vector": vec![json!(1); 4],
        "public_tokens": public_tokens(&staged, active_seat),
        "metadata": {
            "source": "greedy_policy_no_search_corpus",
            "root_seed_u64": seed_u64,
            "ply_index_for_seed": ply_index,
            "completed_turns": game.completed_turns(),
            "turns_remaining": game.turns_remaining(),
            "retained_action_count": action_count,
            "max_actions": args.max_actions,
            "rollouts_per_action": 0,
            "rollout_top_k": args.rollout_top_k,
            "truncated_rollout_samples": 0,
            "prelude_replace_three_of_a_kind": prelude.replace_three_of_a_kind,
            "prelude_wildlife_wipe_count": prelude.wildlife_wipes.len(),
            "teacher": "one_step_greedy_ranker_no_search",
            "scientific_eligibility": "behavior_clone_pretraining"
        },
    });
    Ok((record, selected_action_id))
}

fn export_seed_records(args: &Args, seed_u64: u64) -> Result<Vec<Value>> {
    let config = GameConfig::research_aaaaa(args.player_count)?;
    let mut game = GameState::new(config, GameSeed::from_u64(seed_u64))
        .with_context(|| format!("creating seed {seed_u64}"))?;
    let mut records = Vec::new();
    for ply_index in 0..args.plies_per_seed {
        if game.is_game_over() {
            break;
        }
        let root_record = build_root_record(&game, seed_u64, ply_index, args)
            .with_context(|| format!("seed {seed_u64} ply {ply_index}"))?;
        let selected_action_id = root_record
            .get("selected_action")
            .and_then(Value::as_str)
            .context("selected action missing from generated root")?
            .to_owned();
        game = advance_selected_action(&game, args.max_actions, &selected_action_id)?;
        records.push(root_record);
    }
    Ok(records)
}

fn build_root_record(
    game: &GameState,
    seed_u64: u64,
    ply_index: usize,
    args: &Args,
) -> Result<Value> {
    let active_seat = game.current_player();
    let (prelude, staged) = game.preview_free_three_of_a_kind_if_feasible()?;
    let candidates =
        rank_greedy_actions(&staged, &MarketPrelude::default(), Some(args.max_actions))?;
    if candidates.is_empty() {
        bail!("no legal greedy candidates for non-terminal root");
    }
    let afterstates = candidate_afterstates(&staged, &candidates, active_seat)?;
    let exact_afterstate_scores: Vec<f64> = afterstates
        .iter()
        .map(|afterstate| afterstate.exact_score_active)
        .collect();

    let rollouts = evaluate_candidate_rollouts(
        &staged,
        &afterstates,
        active_seat,
        seed_u64,
        ply_index,
        args.max_actions,
        args.rollouts_per_action,
        args.rollout_top_k,
        args.rollout_determinize,
    )?;

    let selected_index = best_rollout_index(&rollouts, |_| true)?;
    let selected_scores = &rollouts[selected_index].score_means;
    let q_values: Vec<f64> = rollouts
        .iter()
        .map(|rollout| rollout.active_score)
        .collect();
    let q_variances: Vec<f64> = rollouts
        .iter()
        .map(|rollout| rollout.active_score_variance)
        .collect();
    let priors = softmax(&q_values, SOFTMAX_TEMPERATURE);
    let selected_action_id = action_id(&rollouts[selected_index].candidate.action)?;
    let legal_actions = rollouts
        .iter()
        .enumerate()
        .map(|(index, rollout)| {
            action_token(
                &staged,
                &prelude,
                &rollout.candidate.action,
                rollout.candidate.resulting_base_score,
                active_seat,
                index,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let action_count = legal_actions.len();
    let final_score_vector: Vec<Value> = selected_scores
        .iter()
        .map(|score| json!(score.total))
        .collect();
    let score_decomposition = score_decomposition(selected_scores);
    let rank_vector: Vec<Value> = rank_vector(selected_scores)
        .into_iter()
        .map(|rank| json!(rank))
        .collect();
    let visits: Vec<Value> = rollouts
        .iter()
        .map(|rollout| json!(rollout.sample_count))
        .collect();
    let per_action_q: Vec<Value> = q_values.iter().map(|value| json!(value)).collect();
    let exact_afterstate_score_active: Vec<Value> = exact_afterstate_scores
        .iter()
        .map(|value| json!(value))
        .collect();
    let per_action_score_to_go: Vec<Value> = q_values
        .iter()
        .zip(exact_afterstate_scores.iter())
        .map(|(q_value, afterstate)| json!(q_value - afterstate))
        .collect();
    let per_action_q_variance: Vec<Value> = q_variances.iter().map(|value| json!(value)).collect();
    let per_action_q_count: Vec<Value> = rollouts
        .iter()
        .map(|rollout| json!(rollout.sample_count))
        .collect();
    let per_action_truncated_count: Vec<Value> = rollouts
        .iter()
        .map(|rollout| json!(rollout.truncated_count))
        .collect();
    let truncated_rollout_samples: usize =
        rollouts.iter().map(|rollout| rollout.truncated_count).sum();
    let public_hash = staged.public_state().canonical_hash();

    let mut record = json!({
        "schema_id": SCHEMA_ID,
        "state_hash": format!("blake3:{}", public_hash.to_hex()),
        "active_seat": active_seat,
        "legal_actions": legal_actions,
        "priors": priors,
        "visits": visits,
        "per_action_Q": per_action_q,
        "per_action_score_to_go": per_action_score_to_go,
        "per_action_Q_variance": per_action_q_variance,
        "per_action_Q_count": per_action_q_count,
        "per_action_truncated_count": per_action_truncated_count,
        "exact_afterstate_score_active": exact_afterstate_score_active,
        "per_action_Q_valid": vec![json!(true); action_count],
        "selected_action": selected_action_id,
        "chance_samples": [],
        "final_score_vector": final_score_vector,
        "score_decomposition": score_decomposition,
        "rank_vector": rank_vector,
        "public_tokens": public_tokens(&staged, active_seat),
        "metadata": {
            "source": "canonical_simulator_greedy_rollout_dry_run",
            "root_seed_u64": seed_u64,
            "ply_index_for_seed": ply_index,
            "completed_turns": game.completed_turns(),
            "turns_remaining": game.turns_remaining(),
            "retained_action_count": rollouts.len(),
            "max_actions": args.max_actions,
            "rollouts_per_action": args.rollouts_per_action,
            "rollout_top_k": args.rollout_top_k,
            "truncated_rollout_samples": truncated_rollout_samples,
            "prelude_replace_three_of_a_kind": prelude.replace_three_of_a_kind,
            "prelude_wildlife_wipe_count": prelude.wildlife_wipes.len(),
            "teacher": if args.rollouts_per_action == 1 && args.rollout_top_k == 1 {
                "one_greedy_rollout_per_retained_action"
            } else {
                "sampled_topk_greedy_rollout_mean_per_retained_action"
            },
            "scientific_eligibility": "dry_run"
        },
    });
    attach_checksum(&mut record)?;
    Ok(record)
}

fn exact_afterstate_scores_for_candidates(
    staged: &GameState,
    candidates: &[GreedyCandidate],
    active_seat: usize,
) -> Result<Vec<f64>> {
    candidates
        .iter()
        .enumerate()
        .map(|(candidate_index, candidate)| {
            let mut after = staged.clone();
            after.apply(&candidate.action).with_context(|| {
                format!("applying candidate {candidate_index} for exact afterstate score")
            })?;
            let scores = score_game(&after);
            Ok(f64::from(scores[active_seat].total))
        })
        .collect()
}

/// One clone+apply per candidate, reused for exact afterstate scoring and as
/// the rollout/search base state. Candidates whose apply hits a rollout
/// truncation rule (empty bag/stack) are kept with `apply_truncated` set; the
/// partially staged state then scores as its own terminal.
#[derive(Debug, Clone)]
struct CandidateAfterstate {
    candidate: GreedyCandidate,
    state: GameState,
    exact_score_active: f64,
    apply_truncated: bool,
}

fn candidate_afterstates(
    staged: &GameState,
    candidates: &[GreedyCandidate],
    active_seat: usize,
) -> Result<Vec<CandidateAfterstate>> {
    candidates
        .iter()
        .enumerate()
        .map(|(candidate_index, candidate)| {
            let mut after = staged.clone();
            let mut apply_truncated = false;
            if let Err(error) = after.apply(&candidate.action) {
                if is_rollout_truncation_rule_error(&error) {
                    apply_truncated = true;
                } else {
                    return Err(error).with_context(|| {
                        format!("applying candidate {candidate_index} for afterstate")
                    });
                }
            }
            let scores = score_game(&after);
            Ok(CandidateAfterstate {
                candidate: candidate.clone(),
                state: after,
                exact_score_active: f64::from(scores[active_seat].total),
                apply_truncated,
            })
        })
        .collect()
}

/// Public-information-legal afterstate: hidden stack/bag order is resampled
/// before the root action is applied, so neither the action's own market
/// refill nor the continuation can observe the true hidden order.
fn determinized_afterstate(
    staged: &GameState,
    action: &TurnAction,
    determinization_seed: u64,
) -> Result<(GameState, bool)> {
    let mut sim = staged.clone();
    sim.redeterminize_hidden(GameSeed::from_u64(
        determinization_seed ^ HIDDEN_DETERMINIZATION_SALT,
    ));
    let mut truncated = false;
    if let Err(error) = sim.apply(action) {
        if is_rollout_truncation_rule_error(&error) {
            truncated = true;
        } else {
            return Err(error).context("applying root action after hidden determinization");
        }
    }
    Ok((sim, truncated))
}

fn evaluate_candidate_rollouts(
    staged: &GameState,
    afterstates: &[CandidateAfterstate],
    active_seat: usize,
    seed_u64: u64,
    ply_index: usize,
    max_actions: usize,
    rollouts_per_action: usize,
    rollout_top_k: usize,
    determinize: bool,
) -> Result<Vec<ActionRollout>> {
    let mut rollouts = Vec::with_capacity(afterstates.len());
    for (candidate_index, afterstate) in afterstates.iter().enumerate() {
        let mut score_samples = Vec::with_capacity(rollouts_per_action);
        let mut active_samples = Vec::with_capacity(rollouts_per_action);
        let mut truncated_count = 0usize;
        for rollout_index in 0..rollouts_per_action {
            let rollout_seed = rollout_seed(seed_u64, ply_index, candidate_index, rollout_index);
            let mut rng = ChaCha8Rng::seed_from_u64(rollout_seed);
            let (next, mut truncated) = if determinize {
                determinized_afterstate(staged, &afterstate.candidate.action, rollout_seed)?
            } else {
                (afterstate.state.clone(), afterstate.apply_truncated)
            };
            let terminal = if truncated {
                next
            } else {
                let (terminal, continuation_truncated) = complete_with_sampled_greedy(
                    next,
                    max_actions,
                    rollout_top_k,
                    &mut rng,
                    None,
                )?;
                truncated = continuation_truncated;
                terminal
            };
            truncated_count += usize::from(truncated);
            let terminal_scores = score_game(&terminal);
            active_samples.push(f64::from(terminal_scores[active_seat].total));
            score_samples.push(terminal_scores);
        }
        let (active_score, active_score_variance) = mean_variance(&active_samples);
        let sample_count = active_samples.len();
        let score_means = mean_scores(&score_samples);
        rollouts.push(ActionRollout {
            candidate: afterstate.candidate.clone(),
            active_score,
            active_score_variance,
            sample_count,
            truncated_count,
            score_means,
        });
    }
    Ok(rollouts)
}

fn best_rollout_index(
    rollouts: &[ActionRollout],
    allow: impl Fn(&ActionRollout) -> bool,
) -> Result<usize> {
    rollouts
        .iter()
        .enumerate()
        .filter(|(_, rollout)| allow(rollout))
        .max_by(|(_, left), (_, right)| {
            left.active_score
                .partial_cmp(&right.active_score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    right
                        .candidate
                        .immediate_rank
                        .cmp(&left.candidate.immediate_rank)
                })
        })
        .map(|(index, _)| index)
        .context("no allowed rollout candidates")
}

fn run_interactive_policy_game(args: &Args) -> Result<()> {
    let seed_u64 = args.first_seed;
    let config = GameConfig::research_aaaaa(args.player_count)?;
    let mut game = GameState::new(config, GameSeed::from_u64(seed_u64))
        .with_context(|| format!("creating interactive seed {seed_u64}"))?;
    let stdin = std::io::stdin();
    let mut input = stdin.lock();
    let mut stdout = std::io::stdout();
    let mut decision_records = Vec::new();
    let started = std::time::Instant::now();

    while !game.is_game_over() {
        let ply_index = usize::from(game.completed_turns());
        let root_bundle = build_interactive_root(&game, seed_u64, ply_index, args)?;
        let root_message = json!({
            "type": "root",
            "seed_u64": seed_u64,
            "ply_index": ply_index,
            "active_seat": root_bundle.active_seat,
            "candidate_count": root_bundle.candidates.len(),
            "root": root_bundle.record,
        });
        writeln!(stdout, "{}", canonical_json(&root_message))?;
        stdout.flush()?;

        let mut request_line = String::new();
        let bytes = input.read_line(&mut request_line)?;
        if bytes == 0 {
            bail!("interactive policy stream ended before game completed");
        }
        let request: Value =
            serde_json::from_str(&request_line).context("invalid policy request JSON")?;
        if request.get("type").and_then(Value::as_str) == Some("stop") {
            bail!("interactive policy requested stop at ply {ply_index}");
        }
        let decision_started = std::time::Instant::now();
        let decision = apply_policy_request(
            &root_bundle.staged,
            &root_bundle.candidates,
            &request,
            seed_u64,
            ply_index,
            args,
        )?;
        game = root_bundle.staged;
        game.apply(&decision.selected_action)
            .context("applying interactive selected action")?;
        let decision_seconds = decision_started.elapsed().as_secs_f64();
        let decision_record = json!({
            "type": "decision",
            "seed_u64": seed_u64,
            "ply_index": ply_index,
            "active_seat": root_bundle.active_seat,
            "candidate_count": root_bundle.candidates.len(),
            "retained_count": decision.retained_count,
            "selected_action_id": decision.selected_action_id,
            "selected_active_score": decision.selected_active_score,
            "full_best_action_id": decision.full_best_action_id,
            "full_best_active_score": decision.full_best_active_score,
            "full_best_retained": decision.full_best_retained,
            "search_regret": decision.search_regret,
            "decision_seconds": decision_seconds,
            "rollouts_per_action": args.rollouts_per_action,
            "rollout_top_k": args.rollout_top_k,
        });
        writeln!(stdout, "{}", canonical_json(&decision_record))?;
        stdout.flush()?;
        decision_records.push(decision_record);
    }

    let scores = score_game(&game);
    let done_message = json!({
        "type": "done",
        "seed_u64": seed_u64,
        "turns": game.completed_turns(),
        "scores": scores.iter().map(score_breakdown_json).collect::<Vec<_>>(),
        "decision_count": decision_records.len(),
        "elapsed_seconds": started.elapsed().as_secs_f64(),
        "final_state_hash": format!("blake3:{}", game.canonical_hash().to_hex()),
        "scientific_eligibility": "interactive_prefilter_game_pilot",
    });
    writeln!(stdout, "{}", canonical_json(&done_message))?;
    stdout.flush()?;
    Ok(())
}

struct InteractiveRoot {
    record: Value,
    staged: GameState,
    candidates: Vec<GreedyCandidate>,
    active_seat: usize,
}

fn build_interactive_root(
    game: &GameState,
    seed_u64: u64,
    ply_index: usize,
    args: &Args,
) -> Result<InteractiveRoot> {
    let active_seat = game.current_player();
    let (prelude, staged) = game.preview_free_three_of_a_kind_if_feasible()?;
    let candidates =
        rank_greedy_actions(&staged, &MarketPrelude::default(), Some(args.max_actions))?;
    if candidates.is_empty() {
        bail!("no legal candidates for interactive root");
    }
    let exact_afterstate_scores =
        exact_afterstate_scores_for_candidates(&staged, &candidates, active_seat)?;
    let legal_actions = candidates
        .iter()
        .enumerate()
        .map(|(index, candidate)| {
            action_token(
                &staged,
                &prelude,
                &candidate.action,
                candidate.resulting_base_score,
                active_seat,
                index,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let action_count = legal_actions.len();
    let current_scores = score_game(game);
    let score_means = current_scores
        .iter()
        .map(|score| ScoreMean {
            wildlife: f64::from(score.wildlife.iter().sum::<u16>()),
            habitat: f64::from(score.habitat.iter().sum::<u16>()),
            nature_tokens: f64::from(score.nature_tokens),
            total: f64::from(score.total),
        })
        .collect::<Vec<_>>();
    let uniform_prior = 1.0 / action_count as f64;
    let public_hash = staged.public_state().canonical_hash();
    let mut record = json!({
        "schema_id": SCHEMA_ID,
        "state_hash": format!("blake3:{}", public_hash.to_hex()),
        "active_seat": active_seat,
        "legal_actions": legal_actions,
        "priors": vec![json!(uniform_prior); action_count],
        "visits": vec![json!(0); action_count],
        "per_action_Q": vec![json!(0.0); action_count],
        "per_action_score_to_go": vec![json!(0.0); action_count],
        "per_action_Q_variance": vec![json!(0.0); action_count],
        "per_action_Q_count": vec![json!(1.0); action_count],
        "per_action_truncated_count": vec![json!(0); action_count],
        "per_action_Q_valid": vec![json!(true); action_count],
        "exact_afterstate_score_active": exact_afterstate_scores.iter().map(|score| json!(score)).collect::<Vec<_>>(),
        "selected_action": action_id(&candidates[0].action)?,
        "chance_samples": [],
        "final_score_vector": score_means.iter().map(|score| json!(score.total)).collect::<Vec<_>>(),
        "score_decomposition": score_decomposition(&score_means),
        "rank_vector": rank_vector(&score_means).into_iter().map(|rank| json!(rank)).collect::<Vec<_>>(),
        "public_tokens": public_tokens(&staged, active_seat),
        "metadata": {
            "source": "interactive_policy_game_root",
            "root_seed_u64": seed_u64,
            "ply_index_for_seed": ply_index,
            "completed_turns": game.completed_turns(),
            "turns_remaining": game.turns_remaining(),
            "retained_action_count": action_count,
            "max_actions": args.max_actions,
            "rollouts_per_action": 0,
            "rollout_top_k": args.rollout_top_k,
            "prelude_replace_three_of_a_kind": prelude.replace_three_of_a_kind,
            "prelude_wildlife_wipe_count": prelude.wildlife_wipes.len(),
            "teacher": "none_interactive_inference_root",
            "scientific_eligibility": "interactive_prefilter_game_pilot"
        },
    });
    attach_checksum(&mut record)?;
    Ok(InteractiveRoot {
        record,
        staged,
        candidates,
        active_seat,
    })
}

struct PolicyDecision {
    selected_action: TurnAction,
    selected_action_id: String,
    selected_active_score: Option<f64>,
    retained_count: usize,
    full_best_action_id: Option<String>,
    full_best_active_score: Option<f64>,
    full_best_retained: Option<bool>,
    search_regret: Option<f64>,
}

fn apply_policy_request(
    staged: &GameState,
    candidates: &[GreedyCandidate],
    request: &Value,
    seed_u64: u64,
    ply_index: usize,
    args: &Args,
) -> Result<PolicyDecision> {
    let action_ids = candidates
        .iter()
        .map(|candidate| action_id(&candidate.action))
        .collect::<Result<Vec<_>>>()?;
    if let Some(action_id_value) = request.get("action_id").and_then(Value::as_str) {
        let Some((index, _)) = action_ids
            .iter()
            .enumerate()
            .find(|(_, candidate_id)| candidate_id.as_str() == action_id_value)
        else {
            bail!("requested action id {action_id_value} is not legal in current root");
        };
        return Ok(PolicyDecision {
            selected_action: candidates[index].action.clone(),
            selected_action_id: action_id_value.to_owned(),
            selected_active_score: None,
            retained_count: 1,
            full_best_action_id: None,
            full_best_active_score: None,
            full_best_retained: None,
            search_regret: None,
        });
    }

    let retained_ids = request
        .get("retain_action_ids")
        .and_then(Value::as_array)
        .context("policy request must contain action_id or retain_action_ids")?
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(str::to_owned)
                .context("retain_action_ids entries must be strings")
        })
        .collect::<Result<Vec<_>>>()?;
    if retained_ids.is_empty() {
        bail!("retain_action_ids must not be empty");
    }
    let retained_set = retained_ids.iter().cloned().collect::<HashSet<_>>();
    let unknown = retained_set
        .iter()
        .filter(|candidate_id| !action_ids.iter().any(|known| known == *candidate_id))
        .cloned()
        .collect::<Vec<_>>();
    if !unknown.is_empty() {
        bail!("retain_action_ids includes non-legal actions: {unknown:?}");
    }

    let shadow_full = request
        .get("shadow_full_search")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let retained_candidates = if shadow_full {
        candidates.to_vec()
    } else {
        candidates
            .iter()
            .zip(action_ids.iter())
            .filter(|(_, candidate_id)| retained_set.contains(*candidate_id))
            .map(|(candidate, _)| candidate.clone())
            .collect::<Vec<_>>()
    };
    let retained_afterstates =
        candidate_afterstates(staged, &retained_candidates, staged.current_player())?;
    let rollouts = evaluate_candidate_rollouts(
        staged,
        &retained_afterstates,
        staged.current_player(),
        seed_u64,
        ply_index,
        args.max_actions,
        args.rollouts_per_action.max(1),
        args.rollout_top_k,
        args.rollout_determinize,
    )?;
    let rollout_action_ids = rollouts
        .iter()
        .map(|rollout| action_id(&rollout.candidate.action))
        .collect::<Result<Vec<_>>>()?;
    let selected_index = best_rollout_index(&rollouts, |rollout| {
        action_id(&rollout.candidate.action)
            .map(|candidate_id| retained_set.contains(&candidate_id))
            .unwrap_or(false)
    })?;
    let selected = &rollouts[selected_index];
    let selected_action_id = action_id(&selected.candidate.action)?;

    let (full_best_action_id, full_best_active_score, full_best_retained, search_regret) =
        if shadow_full {
            let full_index = best_rollout_index(&rollouts, |_| true)?;
            let full = &rollouts[full_index];
            let full_action_id = action_id(&full.candidate.action)?;
            (
                Some(full_action_id.clone()),
                Some(full.active_score),
                Some(retained_set.contains(&full_action_id)),
                Some(full.active_score - selected.active_score),
            )
        } else {
            let _ = rollout_action_ids;
            (None, None, None, None)
        };

    Ok(PolicyDecision {
        selected_action: selected.candidate.action.clone(),
        selected_action_id,
        selected_active_score: Some(selected.active_score),
        retained_count: retained_ids.len(),
        full_best_action_id,
        full_best_active_score,
        full_best_retained,
        search_regret,
    })
}

fn score_breakdown_json(score: &ScoreBreakdown) -> Value {
    json!({
        "wildlife": score.wildlife.iter().map(|value| json!(value)).collect::<Vec<_>>(),
        "habitat": score.habitat.iter().map(|value| json!(value)).collect::<Vec<_>>(),
        "nature_tokens": score.nature_tokens,
        "base_total": score.base_total,
        "total": score.total,
    })
}

fn advance_selected_action(
    game: &GameState,
    max_actions: usize,
    selected_action_id: &str,
) -> Result<GameState> {
    let (_prelude, staged) = game.preview_free_three_of_a_kind_if_feasible()?;
    let candidates = rank_greedy_actions(&staged, &MarketPrelude::default(), Some(max_actions))?;
    for candidate in candidates {
        if action_id(&candidate.action)? == selected_action_id {
            let mut next = staged;
            next.apply(&candidate.action)?;
            return Ok(next);
        }
    }
    bail!("selected action id {selected_action_id} was not reproducible");
}

/// Plays sampled top-k greedy plies until terminal, a truncation rule error,
/// or `max_plies` plies (when set). Callers distinguish a ply-capped stop from
/// a true terminal via `state.is_game_over()`.
fn complete_with_sampled_greedy(
    mut game: GameState,
    max_actions: usize,
    rollout_top_k: usize,
    rng: &mut ChaCha8Rng,
    max_plies: Option<usize>,
) -> Result<(GameState, bool)> {
    let mut guard = 0usize;
    let mut truncated = false;
    if max_plies == Some(0) {
        return Ok((game, truncated));
    }
    while !game.is_game_over() {
        let (_prelude, staged) = game.preview_free_three_of_a_kind_if_feasible()?;
        let candidates =
            match rank_greedy_actions(&staged, &MarketPrelude::default(), Some(max_actions)) {
                Ok(candidates) => candidates,
                Err(SimulationError::Rules(error)) if is_rollout_truncation_rule_error(&error) => {
                    truncated = true;
                    break;
                }
                Err(error) => return Err(error).context("ranking sampled greedy rollout actions"),
            };
        if candidates.is_empty() {
            bail!(
                "no legal action before game over at turn {}",
                game.completed_turns()
            );
        }
        let sample_limit = rollout_top_k.min(candidates.len());
        let sampled_index = if sample_limit == 1 {
            0
        } else {
            rng.gen_range(0..sample_limit)
        };
        let best = &candidates[sampled_index];
        game = staged;
        if let Err(error) = game.apply(&best.action) {
            if is_rollout_truncation_rule_error(&error) {
                truncated = true;
                break;
            }
            return Err(error).context("applying sampled greedy rollout action");
        }
        guard += 1;
        if let Some(cap) = max_plies {
            if guard >= cap {
                break;
            }
        }
        if guard > 100 {
            bail!("greedy rollout exceeded expected turn guard");
        }
    }
    Ok((game, truncated))
}

fn is_rollout_truncation_rule_error(error: &RuleError) -> bool {
    matches!(
        error,
        RuleError::WildlifeBagEmpty | RuleError::TileStackEmpty
    )
}

fn rollout_seed(
    seed_u64: u64,
    ply_index: usize,
    candidate_index: usize,
    rollout_index: usize,
) -> u64 {
    let mut value = seed_u64 ^ 0x9e37_79b9_7f4a_7c15;
    value ^= (ply_index as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= (candidate_index as u64).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^= (rollout_index as u64).wrapping_mul(0x2545_f491_4f6c_dd1d);
    value
}

fn mean_variance(values: &[f64]) -> (f64, f64) {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| {
            let delta = value - mean;
            delta * delta
        })
        .sum::<f64>()
        / values.len() as f64;
    (mean, variance)
}

fn mean_scores(samples: &[Vec<ScoreBreakdown>]) -> Vec<ScoreMean> {
    let player_count = samples
        .first()
        .map(|scores| scores.len())
        .expect("at least one rollout sample");
    let mut means = vec![
        ScoreMean {
            wildlife: 0.0,
            habitat: 0.0,
            nature_tokens: 0.0,
            total: 0.0,
        };
        player_count
    ];
    for scores in samples {
        for (seat, score) in scores.iter().enumerate() {
            means[seat].wildlife += f64::from(score.wildlife.iter().sum::<u16>());
            means[seat].habitat += f64::from(score.habitat.iter().sum::<u16>());
            means[seat].nature_tokens += f64::from(score.nature_tokens);
            means[seat].total += f64::from(score.total);
        }
    }
    let scale = samples.len() as f64;
    for mean in &mut means {
        mean.wildlife /= scale;
        mean.habitat /= scale;
        mean.nature_tokens /= scale;
        mean.total /= scale;
    }
    means
}

fn action_token(
    game: &GameState,
    prelude: &MarketPrelude,
    action: &TurnAction,
    immediate_base_score: u16,
    active_seat: usize,
    placement_id: usize,
) -> Result<Value> {
    let (tile_slot, wildlife_slot, nature_spend) = draft_slots(action.draft);
    let (tile, wildlife) = market_components(game, tile_slot, wildlife_slot)?;
    let target_coord_ref = coord_ref(action.tile.coord, active_seat, placement_id);
    let wildlife_coord_ref = action
        .wildlife
        .map(|coord| coord_ref(coord, active_seat, placement_id))
        .unwrap_or_else(|| coord_ref(action.tile.coord, active_seat, placement_id));
    Ok(json!({
        "action_id": action_id(action)?,
        "active_seat": active_seat,
        "cleanup_choice": cleanup_choice(prelude),
        "nature_spend": nature_spend,
        "draft_slot": tile_slot.index(),
        "tile_slot": tile_slot.index(),
        "wildlife_slot": wildlife_slot.index(),
        "tile_id": tile.id.0,
        "tile_terrain_a": tile.terrain_a as u8,
        "tile_terrain_b": tile.terrain_b.map_or(-1_i16, |terrain| terrain as i16),
        "tile_wildlife_mask": tile.wildlife.bits(),
        "tile_keystone": tile.keystone,
        "wildlife_species": wildlife as u8,
        "tile_ref": format!("tile:{}@slot:{}", tile.id.0, tile_slot.index()),
        "wildlife_ref": format!("{wildlife:?}@slot:{}", wildlife_slot.index()),
        "target_coord_ref": target_coord_ref,
        "rotation": action.tile.rotation.get(),
        "wildlife_coord_ref": wildlife_coord_ref,
        "wildlife_placement_present": action.wildlife.is_some(),
        "raw_draft": format!("{:?}", action.draft),
        "immediate_pre_rollout_base_score": immediate_base_score,
    }))
}

fn public_tokens(game: &GameState, active_seat: usize) -> Value {
    let current_scores = score_game(game);
    let public_supply = game.public_supply();
    let mut tokens = Vec::new();
    let mut relations = Vec::new();
    let mut placed_token_by_coord: HashMap<(usize, i8, i8), usize> = HashMap::new();
    let mut market_tile_tokens = [None; 4];
    let mut market_wildlife_tokens = [None; 4];

    for (seat, board) in game.boards().iter().enumerate() {
        let score = current_scores[seat];
        let token_index = tokens.len();
        tokens.push(json!({
            "token_index": token_index,
            "token_id": format!("player:{seat}"),
            "token_kind": "player",
            "owner_seat": seat,
            "relative_seat": relative_seat(active_seat, seat),
            "nature_tokens": board.nature_tokens(),
            "tile_count": board.tile_count(),
            "current_base_score": score.base_total,
            "current_total_score": score.total,
            "current_wildlife_total": score.wildlife.iter().sum::<u16>(),
            "current_habitat_total": score.habitat.iter().sum::<u16>(),
        }));
    }

    for (seat, board) in game.boards().iter().enumerate() {
        let mut placed_tiles = board.placed_tiles().collect::<Vec<_>>();
        placed_tiles.sort_by_key(|(coord, _)| (coord.q, coord.r));
        for (placement_id, (coord, placed)) in placed_tiles.into_iter().enumerate() {
            let token_index = tokens.len();
            placed_token_by_coord.insert((seat, coord.q, coord.r), token_index);
            tokens.push(json!({
                "token_index": token_index,
                "token_id": format!("placed_tile:{seat}:{}:{}", coord.q, coord.r),
                "token_kind": "placed_tile",
                "owner_seat": seat,
                "relative_seat": relative_seat(active_seat, seat),
                "coord_ref": coord_ref(coord, seat, placement_id),
                "tile_id": placed.tile.id.0,
                "terrain_a": placed.tile.terrain_a as u8,
                "terrain_b": placed.tile.terrain_b.map_or(-1_i16, |terrain| terrain as i16),
                "wildlife_mask": placed.tile.wildlife.bits(),
                "keystone": placed.tile.keystone,
                "rotation": placed.rotation.get(),
                "placed_wildlife": placed.wildlife.map_or(-1_i16, |wildlife| wildlife as i16),
            }));
        }
    }

    for (seat, board) in game.boards().iter().enumerate() {
        for (coord, placed) in board.placed_tiles() {
            let Some(&source) = placed_token_by_coord.get(&(seat, coord.q, coord.r)) else {
                continue;
            };
            for edge in 0..6 {
                let neighbor_coord = coord.neighbor(edge);
                let Some(neighbor) = board.tile_at(neighbor_coord) else {
                    continue;
                };
                let Some(&target) =
                    placed_token_by_coord.get(&(seat, neighbor_coord.q, neighbor_coord.r))
                else {
                    continue;
                };
                let terrain_matches = placed.tile.terrain_on_edge(placed.rotation, edge)
                    == neighbor
                        .tile
                        .terrain_on_edge(neighbor.rotation, (edge + 3) % 6);
                relations.push(json!({
                    "source": source,
                    "target": target,
                    "relation_kind": "adjacent_hex",
                    "owner_seat": seat,
                    "direction": edge,
                    "terrain_matches": terrain_matches,
                }));
            }
        }
    }

    for (seat, board) in game.boards().iter().enumerate() {
        let mut frontier = board.frontier();
        frontier.sort_unstable();
        for (placement_id, coord) in frontier.into_iter().enumerate() {
            let neighbor_count = coord
                .neighbors()
                .into_iter()
                .filter(|neighbor| board.tile_at(*neighbor).is_some())
                .count();
            let token_index = tokens.len();
            tokens.push(json!({
                "token_index": token_index,
                "token_id": format!("frontier:{seat}:{}:{}", coord.q, coord.r),
                "token_kind": "frontier",
                "owner_seat": seat,
                "relative_seat": relative_seat(active_seat, seat),
                "coord_ref": coord_ref(coord, seat, placement_id),
                "neighbor_count": neighbor_count,
                "active_frontier": seat == active_seat,
            }));
        }
    }

    for slot in MarketSlot::ALL {
        if let Some(tile) = game.market().tiles[slot.index()] {
            let token_index = tokens.len();
            market_tile_tokens[slot.index()] = Some(token_index);
            tokens.push(json!({
                "token_index": token_index,
                "token_id": format!("market_tile:{}", slot.index()),
                "token_kind": "market_tile",
                "market_slot": slot.index(),
                "tile_id": tile.id.0,
                "terrain_a": tile.terrain_a as u8,
                "terrain_b": tile.terrain_b.map_or(-1_i16, |terrain| terrain as i16),
                "wildlife_mask": tile.wildlife.bits(),
                "keystone": tile.keystone,
            }));
        }
        if let Some(wildlife) = game.market().wildlife[slot.index()] {
            let token_index = tokens.len();
            market_wildlife_tokens[slot.index()] = Some(token_index);
            tokens.push(json!({
                "token_index": token_index,
                "token_id": format!("market_wildlife:{}", slot.index()),
                "token_kind": "market_wildlife",
                "market_slot": slot.index(),
                "species": wildlife as u8,
            }));
        }
    }

    for slot in MarketSlot::ALL {
        if let (Some(source), Some(target)) = (
            market_tile_tokens[slot.index()],
            market_wildlife_tokens[slot.index()],
        ) {
            relations.push(json!({
                "source": source,
                "target": target,
                "relation_kind": "same_market_slot",
                "market_slot": slot.index(),
            }));
            relations.push(json!({
                "source": target,
                "target": source,
                "relation_kind": "same_market_slot",
                "market_slot": slot.index(),
            }));
        }
    }

    let supply_token_index = tokens.len();
    tokens.push(json!({
        "token_index": supply_token_index,
        "token_id": "public_supply:0",
        "token_kind": "public_supply",
        "wildlife_bag": public_supply.wildlife_bag,
        "unseen_tile_terrain_capacity": public_supply.unseen_tile_terrain_capacity,
        "unseen_tile_wildlife_capacity": public_supply.unseen_tile_wildlife_capacity,
        "unseen_keystones_by_terrain": public_supply.unseen_keystones_by_terrain,
        "unseen_dual_terrain_pairs": public_supply.unseen_dual_terrain_pairs,
    }));

    json!({
        "schema_id": "cascadiav3.public_tokens.v1",
        "token_count": tokens.len(),
        "relation_count": relations.len(),
        "active_seat": active_seat,
        "tokens": tokens,
        "relations": relations,
    })
}

fn relative_seat(active_seat: usize, seat: usize) -> usize {
    (seat + 4 - active_seat) % 4
}

fn draft_slots(draft: DraftChoice) -> (MarketSlot, MarketSlot, u8) {
    match draft {
        DraftChoice::Paired { slot } => (slot, slot, 0),
        DraftChoice::Independent {
            tile_slot,
            wildlife_slot,
        } => (tile_slot, wildlife_slot, 1),
    }
}

fn market_components(
    game: &GameState,
    tile_slot: MarketSlot,
    wildlife_slot: MarketSlot,
) -> Result<(Tile, Wildlife)> {
    let market = game.market();
    let tile = market.tiles[tile_slot.index()]
        .with_context(|| format!("tile slot {} is unavailable", tile_slot.index()))?;
    let wildlife = market.wildlife[wildlife_slot.index()]
        .with_context(|| format!("wildlife slot {} is unavailable", wildlife_slot.index()))?;
    Ok((tile, wildlife))
}

fn cleanup_choice(prelude: &MarketPrelude) -> String {
    if !prelude.replace_three_of_a_kind && prelude.wildlife_wipes.is_empty() {
        return "none".to_owned();
    }
    let wipe_masks: Vec<String> = prelude
        .wildlife_wipes
        .iter()
        .map(|wipe| {
            wipe.slots
                .iter()
                .map(|slot| slot.index().to_string())
                .collect::<Vec<_>>()
                .join("")
        })
        .collect();
    format!(
        "replace_three_of_a_kind={};wildlife_wipes={}",
        prelude.replace_three_of_a_kind,
        wipe_masks.join("|")
    )
}

fn coord_ref(coord: HexCoord, owner_seat: usize, placement_id: usize) -> Value {
    let q = i32::from(coord.q);
    let r = i32::from(coord.r);
    let s = -q - r;
    let radius6_member = q.abs().max(r.abs()).max(s.abs()) <= 6;
    if radius6_member {
        json!({
            "kind": "canonical",
            "q": q,
            "r": r,
            "s": s,
            "radius6_member": true,
            "cell_index": radius6_cell_index(q, r),
        })
    } else {
        json!({
            "kind": "overflow",
            "q": q,
            "r": r,
            "s": s,
            "radius6_member": false,
            "owner_seat": owner_seat,
            "placement_id": placement_id,
        })
    }
}

fn radius6_cell_index(target_q: i32, target_r: i32) -> usize {
    let mut index = 0usize;
    for radius in 0_i32..=6 {
        for q in -radius..=radius {
            for r in -radius..=radius {
                let s = -q - r;
                if q.abs().max(r.abs()).max(s.abs()) == radius {
                    if q == target_q && r == target_r {
                        return index;
                    }
                    index += 1;
                }
            }
        }
    }
    unreachable!("radius-6 member must appear in the canonical table")
}

fn score_decomposition(scores: &[ScoreMean]) -> Value {
    let mut out = Map::new();
    for (seat, score) in scores.iter().enumerate() {
        out.insert(
            seat.to_string(),
            json!({
                "wildlife": score.wildlife,
                "habitat": score.habitat,
                "nature_tokens": score.nature_tokens,
                "total": score.total,
            }),
        );
    }
    Value::Object(out)
}

fn rank_vector(scores: &[ScoreMean]) -> Vec<u8> {
    scores
        .iter()
        .map(|score| {
            1 + scores
                .iter()
                .filter(|other| other.total > score.total)
                .count() as u8
        })
        .collect()
}

fn softmax(values: &[f64], temperature: f64) -> Vec<Value> {
    let max_value = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = values
        .iter()
        .map(|value| ((value - max_value) / temperature).exp())
        .collect();
    let total: f64 = weights.iter().sum();
    weights
        .into_iter()
        .map(|weight| json!(weight / total))
        .collect()
}

fn action_id(action: &TurnAction) -> Result<String> {
    let bytes = serde_json::to_vec(action)?;
    Ok(format!("sha256:{}", sha256_hex(&bytes)))
}

fn attach_checksum(record: &mut Value) -> Result<()> {
    record
        .as_object_mut()
        .context("record must be a JSON object before checksum")?
        .remove("checksum");
    let canonical = canonical_json(record);
    record
        .as_object_mut()
        .context("record must be a JSON object after checksum")?
        .insert(
            "checksum".to_owned(),
            json!(sha256_hex(canonical.as_bytes())),
        );
    Ok(())
}

fn write_jsonl(path: &PathBuf, records: &[Value]) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut handle = BufWriter::new(File::create(path)?);
    for record in records {
        writeln!(handle, "{}", canonical_json(record))?;
    }
    Ok(())
}

fn write_manifest(path: &PathBuf, records: &[Value], args: &Args) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let manifest = json!({
        "schema_id": SCHEMA_ID,
        "source_generator": "cascadiav3-real-root-exporter",
        "seed_domain": format!(
            "first_seed={},seed_count={},plies_per_seed={},max_actions={},rollouts_per_action={},rollout_top_k={}",
            args.first_seed,
            args.seed_count,
            args.plies_per_seed,
            args.max_actions,
            args.rollouts_per_action,
            args.rollout_top_k
        ),
        "record_count": records.len(),
        "checksum": records_checksum(records),
        "scientific_eligibility": "dry_run",
        "created_at_utc": "2026-06-29T00:00:00+00:00",
        "format": "jsonl",
        "rayon_current_num_threads": rayon::current_num_threads(),
        "notes": format!(
            "Canonical simulator dry-run roots with {} rollout sample(s) per retained legal action and rollout_top_k={}; not strength evidence.",
            args.rollouts_per_action,
            args.rollout_top_k
        ),
    });
    std::fs::write(path, format!("{}\n", canonical_json(&manifest)))?;
    Ok(())
}

fn write_stream_manifest(
    path: &PathBuf,
    args: &Args,
    record_count: usize,
    checksum: &str,
    scientific_eligibility: &str,
    notes: &str,
) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let manifest = json!({
        "schema_id": SCHEMA_ID,
        "source_generator": "cascadiav3-real-root-exporter",
        "seed_domain": format!(
            "first_seed={},seed_count={},plies_per_seed={},max_actions={},mode=greedy_policy_corpus",
            args.first_seed,
            args.seed_count,
            args.plies_per_seed,
            args.max_actions
        ),
        "record_count": record_count,
        "checksum": checksum,
        "scientific_eligibility": scientific_eligibility,
        "created_at_utc": "2026-06-30T00:00:00+00:00",
        "format": "jsonl",
        "rayon_current_num_threads": rayon::current_num_threads(),
        "notes": notes,
    });
    std::fs::write(path, format!("{}\n", canonical_json(&manifest)))?;
    Ok(())
}

fn tensor_shard_metadata(args: &Args, shard: &TensorShardData) -> Value {
    json!({
        "version": SHARD_VERSION,
        "source": "greedy_policy_no_search_corpus",
        "source_paths": ["rust-native:greedy_policy_tensor_corpus"],
        "format": "npz",
        "dtype": "float16",
        "compression": tensor_compression_label(args.tensor_compression),
        "record_count": shard.record_count,
        "total_token_count": shard.total_token_count,
        "total_action_count": shard.total_action_count,
        "token_feature_dim": PUBLIC_TOKEN_FEATURE_DIM,
        "action_feature_dim": SEMANTIC_PUBLIC_TOKEN_ACTION_FEATURE_DIM,
        "max_token_count": shard.max_token_count,
        "max_action_count": shard.max_action_count,
        "first_state_hash": shard.first_state_hash,
        "last_state_hash": shard.last_state_hash,
        "feature_extractor": "rust:cascadiav3-real-root-exporter",
        "feature_extractor_contract": "public_token_features_v1+semantic_public_token_action_features_v1",
        "seed_domain": format!(
            "first_seed={},seed_count={},plies_per_seed={},max_actions={},mode=greedy_policy_tensor_corpus",
            args.first_seed,
            args.seed_count,
            args.plies_per_seed,
            args.max_actions
        ),
        "canonical_model_inputs": [
            "public_token_features",
            "semantic_public_token_action_features",
            "selected_action_index"
        ],
        "omitted_by_design": [
            "raw_legal_action_json",
            "state_hashes",
            "action_ids",
            "relations",
            "per_action_Q",
            "score_decompositions"
        ],
    })
}

fn expert_tensor_shard_metadata(args: &Args, shard: &ExpertTensorShardData) -> Value {
    json!({
        "version": EXPERT_SHARD_VERSION,
        "schema_id": EXPERT_TENSOR_SCHEMA_ID,
        "source": "expert_root_chance_mcts_dry_run_tensor_corpus",
        "source_paths": ["rust-native:expert_tensor_corpus"],
        "format": "npz",
        "dtype": {
            "features": "float16",
            "targets": "float32",
            "relations": "sparse_i32_triples"
        },
        "compression": tensor_compression_label(args.tensor_compression),
        "record_count": shard.record_count,
        "total_token_count": shard.total_token_count,
        "total_action_count": shard.total_action_count,
        "total_relation_edge_count": shard.total_relation_edge_count,
        "token_feature_dim": PUBLIC_TOKEN_FEATURE_DIM,
        "action_feature_dim": SEMANTIC_PUBLIC_TOKEN_ACTION_FEATURE_DIM,
        "max_token_count": shard.max_token_count,
        "max_action_count": shard.max_action_count,
        "max_relation_edge_count": shard.max_relation_edge_count,
        "first_state_hash": shard.first_state_hash,
        "last_state_hash": shard.last_state_hash,
        "feature_extractor": "rust:cascadiav3-real-root-exporter",
        "feature_extractor_contract": "public_token_features_v1+semantic_public_token_action_features_v1+sparse_combined_relation_ids_v1",
        "seed_domain": format!(
            "first_seed={},seed_count={},plies_per_seed={},max_actions={},rollouts_per_action={},rollout_top_k={},mode=expert_tensor_corpus",
            args.first_seed,
            args.seed_count,
            args.plies_per_seed,
            args.max_actions,
            args.rollouts_per_action,
            args.rollout_top_k,
        ),
        "canonical_model_inputs": [
            "public_token_features",
            "semantic_public_token_action_features",
            "sparse_relation_edges"
        ],
        "canonical_targets": [
            "selected_action_index",
            "per_action_Q",
            "per_action_score_to_go",
            "per_action_Q_valid",
            "priors",
            "visits",
            "per_action_Q_variance",
            "per_action_Q_count",
            "per_action_truncated_count",
            "exact_afterstate_score_active",
            "final_score_vector",
            "rank_vector",
            "score_decomposition"
        ],
        "omitted_by_design": [
            "raw_legal_action_json",
            "raw_replay_prefix",
            "action_ids",
            "afterstate_hashes",
            "chance_sample_private_audit_hashes"
        ],
    })
}

fn write_tensor_manifest(
    path: &PathBuf,
    args: &Args,
    shard: &TensorShardData,
    checksum: &str,
) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let manifest = json!({
        "schema_id": GREEDY_TENSOR_SCHEMA_ID,
        "version": SHARD_VERSION,
        "source_generator": "cascadiav3-real-root-exporter",
        "seed_domain": format!(
            "first_seed={},seed_count={},plies_per_seed={},max_actions={},mode=greedy_policy_tensor_corpus",
            args.first_seed,
            args.seed_count,
            args.plies_per_seed,
            args.max_actions
        ),
        "record_count": shard.record_count,
        "checksum": checksum,
        "scientific_eligibility": "behavior_clone_pretraining",
        "created_at_utc": "2026-06-30T00:00:00+00:00",
        "format": "npz",
        "dtype": "float16",
        "compression": tensor_compression_label(args.tensor_compression),
        "token_feature_dim": PUBLIC_TOKEN_FEATURE_DIM,
        "action_feature_dim": SEMANTIC_PUBLIC_TOKEN_ACTION_FEATURE_DIM,
        "total_token_count": shard.total_token_count,
        "total_action_count": shard.total_action_count,
        "max_token_count": shard.max_token_count,
        "max_action_count": shard.max_action_count,
        "rayon_current_num_threads": rayon::current_num_threads(),
        "notes": "Rust-native compact greedy behavior-cloning tensor shard; public-token and semantic action features are extracted in Rust and written directly to compressed .npz.",
    });
    std::fs::write(path, format!("{}\n", canonical_json(&manifest)))?;
    Ok(())
}

fn write_expert_tensor_manifest(
    path: &PathBuf,
    args: &Args,
    shard: &ExpertTensorShardData,
    checksum: &str,
) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let (mode_name, scientific_eligibility, notes) = match args.mode {
        Mode::GreedyExpertTensorCorpus => (
            "greedy_expert_tensor_corpus",
            "behavior_clone_pretraining",
            "Rust-native packed expert tensor shard from greedy self-play; relation edges are present and labels are one-step greedy behavior-cloning targets.",
        ),
        Mode::GreedyStateSearchBootstrapTensorCorpus => (
            "greedy_state_search_bootstrap_tensor_corpus",
            "expert_iteration_bootstrap",
            "Rust-native packed expert tensor shard from greedy-state roots; retained greedy-ranked actions are labeled by sampled greedy rollout means while real trajectories advance by the greedy action.",
        ),
        Mode::ModelStateSearchBootstrapTensorCorpus => (
            "model_state_search_bootstrap_tensor_corpus",
            "expert_iteration_model_state_bootstrap",
            "Rust-native packed expert tensor shard from model-state roots; retained greedy-ranked actions are labeled by sampled greedy rollout means while real trajectories advance by model derived-Q or model prior.",
        ),
        Mode::GumbelSelfplayTensorCorpus => (
            "gumbel_selfplay_tensor_corpus",
            "gumbel_selfplay_expert_iteration",
            "Rust-native packed v2 expert tensor shard from all-seat Gumbel self-play over determinized hidden states; per-action targets are search completed-Q values, improved_policy is the Gumbel policy-improvement target, and value labels are real terminal outcomes.",
        ),
        _ => (
            "expert_tensor_corpus",
            "expert_iteration_bootstrap_tensor_pretraining",
            "Rust-native packed expert tensor shard; JSONL audit can be generated separately, but trainer scale path reads this NPZ directly.",
        ),
    };
    let (schema_id, version) = if args.mode == Mode::GumbelSelfplayTensorCorpus {
        (EXPERT_TENSOR_SCHEMA_ID_V2, EXPERT_SHARD_VERSION_V2)
    } else {
        (EXPERT_TENSOR_SCHEMA_ID, EXPERT_SHARD_VERSION)
    };
    let manifest = json!({
        "schema_id": schema_id,
        "version": version,
        "source_generator": "cascadiav3-real-root-exporter",
        "seed_domain": format!(
            "first_seed={},seed_count={},plies_per_seed={},max_actions={},rollouts_per_action={},rollout_top_k={},mode={}",
            args.first_seed,
            args.seed_count,
            args.plies_per_seed,
            args.max_actions,
            args.rollouts_per_action,
            args.rollout_top_k,
            mode_name,
        ),
        "record_count": shard.record_count,
        "checksum": checksum,
        "scientific_eligibility": scientific_eligibility,
        "created_at_utc": "2026-06-30T00:00:00+00:00",
        "format": "npz",
        "feature_dtype": "float16",
        "target_dtype": "float32",
        "relation_encoding": "sparse_i32_source_target_relation_id",
        "compression": tensor_compression_label(args.tensor_compression),
        "token_feature_dim": PUBLIC_TOKEN_FEATURE_DIM,
        "action_feature_dim": SEMANTIC_PUBLIC_TOKEN_ACTION_FEATURE_DIM,
        "total_token_count": shard.total_token_count,
        "total_action_count": shard.total_action_count,
        "total_relation_edge_count": shard.total_relation_edge_count,
        "max_token_count": shard.max_token_count,
        "max_action_count": shard.max_action_count,
        "max_relation_edge_count": shard.max_relation_edge_count,
        "rayon_current_num_threads": rayon::current_num_threads(),
        "notes": notes,
    });
    std::fs::write(path, format!("{}\n", canonical_json(&manifest)))?;
    Ok(())
}

fn records_checksum(records: &[Value]) -> String {
    let mut digest = Sha256::new();
    for record in records {
        digest.update(canonical_json(record).as_bytes());
        digest.update(b"\n");
    }
    format!("{:x}", digest.finalize())
}

fn read_jsonl(path: &PathBuf) -> Result<Vec<Value>> {
    let handle = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let reader = std::io::BufReader::new(handle);
    let mut records = Vec::new();
    for (line_index, line) in reader.lines().enumerate() {
        let line =
            line.with_context(|| format!("reading {}:{}", path.display(), line_index + 1))?;
        if line.trim().is_empty() {
            continue;
        }
        let record: Value = serde_json::from_str(&line)
            .with_context(|| format!("parsing {}:{}", path.display(), line_index + 1))?;
        records.push(record);
    }
    if records.is_empty() {
        bail!("{} contains no JSONL records", path.display());
    }
    Ok(records)
}

fn verify_record_checksum(record: &Value) -> Result<()> {
    let expected = record
        .get("checksum")
        .and_then(Value::as_str)
        .context("record checksum missing")?;
    let mut without = record.clone();
    without
        .as_object_mut()
        .context("record must be object")?
        .remove("checksum");
    let actual = sha256_hex(canonical_json(&without).as_bytes());
    if actual != expected {
        bail!("record checksum mismatch");
    }
    Ok(())
}

fn game_hash(game: &GameState) -> String {
    format!("blake3:{}", game.canonical_hash().to_hex())
}

fn public_hash(game: &GameState) -> String {
    format!("blake3:{}", game.public_state().canonical_hash().to_hex())
}

fn binary_hash() -> String {
    std::env::current_exe()
        .ok()
        .and_then(|path| sha256_file_hex(&path).ok())
        .map(|hash| format!("sha256:{hash}"))
        .unwrap_or_else(|| "sha256:unavailable".to_owned())
}

fn sha256_file_hex(path: &PathBuf) -> Result<String> {
    let mut file = File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(bytes);
    format!("{:x}", digest.finalize())
}

fn canonical_json(value: &Value) -> String {
    match value {
        Value::Null => "null".to_owned(),
        Value::Bool(value) => value.to_string(),
        Value::Number(value) => value.to_string(),
        Value::String(value) => serde_json::to_string(value).expect("string serialization"),
        Value::Array(values) => {
            let items = values.iter().map(canonical_json).collect::<Vec<_>>();
            format!("[{}]", items.join(","))
        }
        Value::Object(values) => {
            let sorted: BTreeMap<_, _> = values.iter().collect();
            let items = sorted
                .into_iter()
                .map(|(key, value)| {
                    format!(
                        "{}:{}",
                        serde_json::to_string(key).expect("key serialization"),
                        canonical_json(value)
                    )
                })
                .collect::<Vec<_>>();
            format!("{{{}}}", items.join(","))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn test_args() -> Args {
        Args {
            mode: Mode::ExportRoots,
            out: PathBuf::from("/tmp/unused.jsonl"),
            manifest: PathBuf::from("/tmp/unused_manifest.json"),
            first_seed: 2_026_063_000,
            seed_count: 1,
            plies_per_seed: 2,
            max_actions: 4,
            rollouts_per_action: 2,
            rollout_top_k: 2,
            player_count: 4,
            rayon_threads: None,
            tensor_compression: NpzCompression::Stored,
            input: None,
            bench_out: None,
            allow_model_fallback: true,
            model_service: None,
            model_manifest: None,
            model_timeout_ms: 1_000,
            rollout_determinize: false,
            gumbel_n_simulations: 8,
            gumbel_top_m: 4,
            gumbel_depth_rounds: 1,
            gumbel_determinizations: 2,
            gumbel_blend_weight: 1.0,
            gumbel_exploration: false,
            gumbel_max_root_actions: None,
            gumbel_root_menu: 64,
            k_interior: 6,
            model_sessions: None,
            shared_model_session: false,
        }
    }

    fn advanced_test_state(seed_u64: u64, plies: usize) -> GameState {
        let config = GameConfig::research_aaaaa(4).expect("4p config");
        let mut game = GameState::new(config, GameSeed::from_u64(seed_u64)).expect("game");
        let mut rng = ChaCha8Rng::seed_from_u64(seed_u64 ^ 0xabcd);
        for _ in 0..plies {
            let (terminal, _) =
                complete_with_sampled_greedy(game, 4, 2, &mut rng, Some(1)).expect("advance");
            game = terminal;
            if game.is_game_over() {
                break;
            }
        }
        game
    }

    #[test]
    fn determinized_rollouts_never_observe_true_hidden_order() {
        let game = advanced_test_state(2_026_063_100, 6);
        let (_prelude, staged) = game
            .preview_free_three_of_a_kind_if_feasible()
            .expect("staged");
        let candidates =
            rank_greedy_actions(&staged, &MarketPrelude::default(), Some(4)).expect("candidates");
        assert!(!candidates.is_empty());

        // Same public state, permuted hidden stack/bag order.
        let mut permuted = staged.clone();
        permuted.redeterminize_hidden(GameSeed::from_u64(0xdead_beef));
        assert_eq!(
            staged.public_state().canonical_hash(),
            permuted.public_state().canonical_hash(),
            "redeterminization must preserve public state"
        );

        let evaluate = |root: &GameState| -> Vec<f64> {
            let afterstates =
                candidate_afterstates(root, &candidates, root.current_player()).expect("after");
            evaluate_candidate_rollouts(root, &afterstates, root.current_player(), 7, 3, 4, 3, 2, true)
                .expect("rollouts")
                .iter()
                .map(|rollout| rollout.active_score)
                .collect()
        };
        assert_eq!(
            evaluate(&staged),
            evaluate(&permuted),
            "determinized rollout labels must be invariant to true hidden order"
        );
    }

    #[test]
    fn sampled_greedy_respects_ply_cap() {
        let game = advanced_test_state(2_026_063_200, 2);
        let start_turns = game.completed_turns();
        let mut rng = ChaCha8Rng::seed_from_u64(11);
        let (capped, truncated) =
            complete_with_sampled_greedy(game.clone(), 4, 2, &mut rng, Some(3)).expect("capped");
        assert!(!truncated);
        assert!(capped.completed_turns() <= start_turns + 3);
        assert!(!capped.is_game_over());

        let mut rng_zero = ChaCha8Rng::seed_from_u64(11);
        let (unmoved, _) =
            complete_with_sampled_greedy(game.clone(), 4, 2, &mut rng_zero, Some(0))
                .expect("zero cap");
        assert_eq!(unmoved.canonical_hash(), game.canonical_hash());
    }

    #[test]
    fn golden_rollout_labels_are_stable() {
        let args = test_args();
        let records = export_seed_records(&args, args.first_seed).expect("golden seed exports");
        assert_eq!(records.len(), args.plies_per_seed);
        let mut hasher = Sha256::new();
        for record in &records {
            hasher.update(canonical_json(record).as_bytes());
        }
        assert_eq!(
            format!("{:x}", hasher.finalize()),
            "39bffaa912cab1a59fd309fbd3a8efc88210abcc9c50129c6b88ca6eb0dc3e39"
        );
    }

    fn mock_bridge_command() -> String {
        format!(
            "python3 {}/../tests/mock_model_bridge.py",
            env!("CARGO_MANIFEST_DIR")
        )
    }

    fn gumbel_test_args(tempdir: &std::path::Path) -> Args {
        let mut args = test_args();
        args.mode = Mode::GumbelSelfplayTensorCorpus;
        args.out = tempdir.join("gumbel_tiny.npz");
        args.manifest = tempdir.join("gumbel_tiny_manifest.json");
        args.model_service = Some(mock_bridge_command());
        args.plies_per_seed = 3;
        args.gumbel_n_simulations = 4;
        args.gumbel_top_m = 2;
        args.gumbel_depth_rounds = 1;
        args.gumbel_determinizations = 2;
        args.gumbel_blend_weight = 1.0;
        args.gumbel_exploration = true;
        args.gumbel_max_root_actions = None;
        args.k_interior = 3;
        args.model_sessions = Some(1);
        args
    }

    #[test]
    fn gumbel_selfplay_records_roundtrip_into_v2_shard() {
        let tempdir = std::env::temp_dir().join(format!(
            "cascadia-gumbel-test-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&tempdir).expect("tempdir");
        let args = gumbel_test_args(&tempdir);

        let session = model_state_worker_session(&args).expect("mock session");
        assert!(session.is_some(), "mock bridge session must spawn");
        let mut bridge = ChunkBridge::Owned(session);
        let records = play_gumbel_selfplay_seed(&args, 2_026_070_600, &mut bridge)
            .expect("selfplay seed plays");
        assert!(!records.is_empty());

        let mut shared_final_score: Option<Value> = None;
        for record in &records {
            let action_count = record["legal_actions"].as_array().unwrap().len();
            let improved = record["improved_policy"].as_array().unwrap();
            assert_eq!(improved.len(), action_count);
            let policy_sum: f64 = improved.iter().map(|value| value.as_f64().unwrap()).sum();
            assert!((policy_sum - 1.0).abs() < 1e-6, "improved policy sums to 1");
            let visits = record["visits"].as_array().unwrap();
            let q_valid = record["per_action_Q_valid"].as_array().unwrap();
            for (visit, valid) in visits.iter().zip(q_valid.iter()) {
                assert_eq!(
                    visit.as_u64().unwrap() > 0,
                    valid.as_bool().unwrap(),
                    "q_valid must equal visits > 0"
                );
            }
            assert!(record["search_root_value"].as_f64().is_some());
            let final_score = record["final_score_vector"].clone();
            match &shared_final_score {
                None => shared_final_score = Some(final_score),
                Some(existing) => assert_eq!(
                    existing, &final_score,
                    "all records of a seed share the real outcome"
                ),
            }
        }

        let shard = ExpertTensorShardData::from_records(&records).expect("v2 shard");
        assert_eq!(shard.improved_policy_records, shard.record_count);
        assert_eq!(shard.improved_policy.len(), shard.total_action_count);
        assert_eq!(shard.search_root_value.len(), shard.record_count);

        // Full export path writes a v2 npz + manifest.
        let written = export_gumbel_selfplay_tensor_corpus(&args).expect("export");
        assert!(written > 0);
        assert!(args.out.exists());

        // Shared-bridge mode produces the same corpus shape through one
        // aggregated session.
        let mut shared_args = args.clone();
        shared_args.shared_model_session = true;
        shared_args.model_sessions = Some(2);
        shared_args.out = tempdir.join("gumbel_tiny_shared.npz");
        shared_args.manifest = tempdir.join("gumbel_tiny_shared_manifest.json");
        let shared_written =
            export_gumbel_selfplay_tensor_corpus(&shared_args).expect("shared export");
        assert_eq!(shared_written, written);
        let manifest: Value = serde_json::from_str(
            &std::fs::read_to_string(&args.manifest).expect("manifest readable"),
        )
        .expect("manifest json");
        assert_eq!(
            manifest["schema_id"].as_str(),
            Some(EXPERT_TENSOR_SCHEMA_ID_V2)
        );
        assert_eq!(
            manifest["version"].as_str(),
            Some(feature_tensors::EXPERT_SHARD_VERSION_V2)
        );
        let _ = std::fs::remove_dir_all(&tempdir);
    }

    #[test]
    fn gumbel_policy_game_emits_decisions_and_done() {
        let tempdir = std::env::temp_dir().join(format!(
            "cascadia-gumbel-game-test-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&tempdir).expect("tempdir");
        let mut args = gumbel_test_args(&tempdir);
        args.mode = Mode::GumbelPolicyGame;
        args.out = tempdir.join("gumbel_game.jsonl");
        args.seed_count = 1;
        args.gumbel_exploration = false;

        run_gumbel_policy_game(&args).expect("policy game runs");
        let contents = std::fs::read_to_string(&args.out).expect("game jsonl");
        let lines: Vec<Value> = contents
            .lines()
            .map(|line| serde_json::from_str(line).expect("jsonl line"))
            .collect();
        assert!(lines.len() > 1);
        let decisions = lines
            .iter()
            .filter(|line| line["type"] == "gumbel_decision")
            .count();
        let done = lines
            .iter()
            .find(|line| line["type"] == "gumbel_game_done")
            .expect("done record");
        assert!(decisions > 0);
        assert_eq!(done["scores"].as_array().unwrap().len(), 4);
        assert_eq!(done["decision_count"].as_u64().unwrap() as usize, decisions);
        let _ = std::fs::remove_dir_all(&tempdir);
    }

    #[test]
    fn radius6_matches_expected_count() {
        let mut seen = HashMap::new();
        for q in -6_i32..=6 {
            for r in -6_i32..=6 {
                let s = -q - r;
                if q.abs().max(r.abs()).max(s.abs()) <= 6 {
                    let index = radius6_cell_index(q, r);
                    assert!(seen.insert(index, (q, r)).is_none());
                }
            }
        }
        assert_eq!(seen.len(), 127);
    }

    #[test]
    fn canonical_json_sorts_object_keys() {
        let value = json!({"b": 2, "a": [true, {"d": null, "c": "x"}]});
        assert_eq!(
            canonical_json(&value),
            r#"{"a":[true,{"c":"x","d":null}],"b":2}"#
        );
    }

    use model_bridge::parse_model_response;

    #[test]
    fn model_response_parser_preserves_q_vectors() {
        let root = json!({
            "action_ids": ["a0", "a1"],
        });
        let response = json!({
            "type": "eval_response",
            "action_ids": ["a0", "a1"],
            "priors": [0.25, 0.75],
            "q": [88.0, 91.5],
            "score_to_go": [12.0, 14.5],
            "model_fallback": false,
        });
        let eval = parse_model_response(&root, response, false).expect("valid model response");
        assert_eq!(eval.priors, vec![json!(0.25), json!(0.75)]);
        assert_eq!(eval.q, Some(vec![88.0, 91.5]));
        assert_eq!(eval.score_to_go, Some(vec![12.0, 14.5]));

        let bad = json!({
            "type": "eval_response",
            "action_ids": ["a0", "a1"],
            "priors": [0.5, 0.5],
            "q": [1.0],
        });
        assert!(parse_model_response(&root, bad, false).is_err());
    }
}
