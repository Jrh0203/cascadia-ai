#![recursion_limit = "256"]

use std::collections::{BTreeMap, HashMap, HashSet};
mod feature_tensors;
mod gumbel;
mod model_bridge;
mod npz_writer;

use std::fs::File;
use std::io::{BufRead, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use cascadia_game::{
    DraftChoice, GameConfig, GameSeed, GameState, HexCoord, MarketPrelude, MarketSlot, RuleError,
    ScoreBreakdown, Tile, TurnAction, Wildlife, score_game,
};
use cascadia_sim::{
    GreedyCandidate, GreedyRankScratch, SimulationError, rank_greedy_actions,
    rank_greedy_actions_with_market_choice, rank_greedy_actions_with_scratch,
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

use feature_tensors::{
    EXPERT_SHARD_VERSION, EXPERT_SHARD_VERSION_V4, ExpertTensorShardData, PUBLIC_TOKEN_FEATURE_DIM,
    SEMANTIC_PUBLIC_TOKEN_ACTION_FEATURE_DIM, SHARD_VERSION, TensorShardData,
};
use model_bridge::{
    BridgeConfig, ModelEval, ModelServiceSession, SharedBridge, SharedBridgeClient,
    average_model_evals, uniform_model_eval,
};
use npz_writer::NpzCompression;

const SCHEMA_ID: &str = "cascadiav3.pre_gpu.v0";
const EXPERT_ROOT_SCHEMA_ID: &str = "cascadiav3.expert_root.v1";
const EXPERT_TENSOR_SCHEMA_ID: &str = "cascadiav3.expert_tensor_shard.v1";
const EXPERT_TENSOR_SCHEMA_ID_V4: &str = "cascadiav3.expert_tensor_shard.v4";
const GREEDY_TENSOR_SCHEMA_ID: &str = "greedy_policy_tensor_shard_v1";
const ROOT_REPLAY_SCHEMA_ID: &str = "cascadiav3.root_replay.v1";
const RULESET_ID: &str = "cascadia_research_aaaaa_4p_card_a_no_habitat_bonus_rules_2026_07_09";
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
    source_revision: Option<String>,
    model_timeout_ms: u64,
    rollout_determinize: bool,
    gumbel_n_simulations: usize,
    gumbel_top_m: usize,
    gumbel_depth_rounds: usize,
    gumbel_determinizations: usize,
    gumbel_market_decision_samples: usize,
    gumbel_exact_endgame_turns: usize,
    gumbel_blend_weight: f64,
    gumbel_parallel_leaf_rollouts: bool,
    gumbel_exploration: bool,
    gumbel_peek: bool,
    gumbel_table_total: bool,
    gumbel_table_native_q: bool,
    gumbel_leaf_softmix: Option<f64>,
    gumbel_c_visit: f64,
    gumbel_c_scale: f64,
    gumbel_sigma_norm: gumbel::SigmaNormalization,
    gumbel_paired_rollouts: bool,
    gumbel_ghost_opponents: bool,
    gumbel_q_bias_correction: bool,
    gumbel_lcb_c: f64,
    gumbel_refresh_sample_divisor: usize,
    gumbel_tta: u8,
    gumbel_max_root_actions: Option<usize>,
    /// Root menu enumeration cap (immediate-score-ranked pre-filter before
    /// the model-prior top-m). 0 = full legal set. Late-game legal menus can
    /// exceed several thousand compound actions, which both bloats eval
    /// requests and blows up the relation-bias memory (B x A x S x d).
    gumbel_root_menu: usize,
    k_interior: usize,
    model_sessions: Option<usize>,
    /// One shared bridge (one CUDA context) with cross-worker request
    /// batching instead of one bridge per persistent seed worker.
    shared_model_session: bool,
    /// Per-seed JSONL output directory for --gumbel-benchmark-batch.
    output_dir: Option<PathBuf>,
    /// Sampling stride over ledger decisions for --search-stability-probe.
    probe_stride: usize,
    /// Search repeats per (root, variant) for --search-stability-probe.
    probe_repeats: usize,
    /// Total sampled roots cap for --search-stability-probe.
    probe_max_roots: usize,
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
    GumbelSuggestServer,
    GumbelBenchmarkBatch,
    GumbelSelfplayTensorCorpus,
    TableContentionAudit,
    SearchStabilityProbe,
    PuzzleBank,
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

fn score_components(score: &ScoreBreakdown) -> [f64; 3] {
    let wildlife = f64::from(score.wildlife.iter().sum::<u16>());
    let nature_tokens = f64::from(score.nature_tokens);
    // Habitat owns any configured habitat bonus so the three components
    // always sum exactly to the engine's final total. The active research
    // rules disable bonuses, but the tensor contract remains general.
    let habitat = f64::from(score.total) - wildlife - nature_tokens;
    [wildlife, habitat, nature_tokens]
}

fn score_mean(score: &ScoreBreakdown) -> ScoreMean {
    let [wildlife, habitat, nature_tokens] = score_components(score);
    ScoreMean {
        wildlife,
        habitat,
        nature_tokens,
        total: f64::from(score.total),
    }
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
        Mode::GumbelSuggestServer => {
            run_gumbel_suggest_server(&args)?;
        }
        Mode::GumbelBenchmarkBatch => {
            run_gumbel_benchmark_batch(&args)?;
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
        Mode::TableContentionAudit => {
            run_table_contention_audit(&args)?;
        }
        Mode::SearchStabilityProbe => {
            run_search_stability_probe(&args)?;
        }
        Mode::PuzzleBank => {
            run_puzzle_bank(&args)?;
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
        source_revision: None,
        model_timeout_ms: 10_000,
        rollout_determinize: false,
        gumbel_n_simulations: 64,
        gumbel_top_m: 16,
        gumbel_depth_rounds: 1,
        gumbel_determinizations: 4,
        gumbel_market_decision_samples: 8,
        gumbel_exact_endgame_turns: 0,
        gumbel_blend_weight: 0.5,
        gumbel_parallel_leaf_rollouts: false,
        gumbel_exploration: false,
        gumbel_peek: false,
        gumbel_table_total: false,
        gumbel_table_native_q: false,
        gumbel_leaf_softmix: None,
        gumbel_c_visit: 50.0,
        gumbel_c_scale: 1.0,
        gumbel_sigma_norm: gumbel::SigmaNormalization::MinMax,
        gumbel_paired_rollouts: false,
        gumbel_ghost_opponents: false,
        gumbel_q_bias_correction: false,
        gumbel_lcb_c: 0.0,
        gumbel_refresh_sample_divisor: 1,
        gumbel_tta: 1,
        gumbel_max_root_actions: None,
        gumbel_root_menu: 256,
        k_interior: 16,
        model_sessions: None,
        shared_model_session: false,
        output_dir: None,
        probe_stride: 7,
        probe_repeats: 6,
        probe_max_roots: 100,
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
            "--gumbel-suggest-server" => args.mode = Mode::GumbelSuggestServer,
            "--gumbel-benchmark-batch" => args.mode = Mode::GumbelBenchmarkBatch,
            "--table-contention-audit" => args.mode = Mode::TableContentionAudit,
            "--search-stability-probe" => args.mode = Mode::SearchStabilityProbe,
            "--puzzle-bank" => args.mode = Mode::PuzzleBank,
            "--probe-stride" => {
                args.probe_stride = value()?.parse().context("invalid --probe-stride")?
            }
            "--probe-repeats" => {
                args.probe_repeats = value()?.parse().context("invalid --probe-repeats")?
            }
            "--probe-max-roots" => {
                args.probe_max_roots = value()?.parse().context("invalid --probe-max-roots")?
            }
            "--output-dir" => args.output_dir = Some(PathBuf::from(value()?)),
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
            "--gumbel-peek" => {
                args.gumbel_peek = true;
            }
            "--gumbel-table-total" => {
                args.gumbel_table_total = true;
            }
            "--gumbel-table-native-q" => {
                args.gumbel_table_native_q = true;
            }
            "--gumbel-leaf-softmix" => {
                args.gumbel_leaf_softmix =
                    Some(value()?.parse().context("invalid --gumbel-leaf-softmix")?);
            }
            "--gumbel-c-visit" => {
                args.gumbel_c_visit = value()?.parse().context("invalid --gumbel-c-visit")?;
                if !(args.gumbel_c_visit >= 0.0) {
                    bail!("--gumbel-c-visit must be non-negative");
                }
            }
            "--gumbel-c-scale" => {
                args.gumbel_c_scale = value()?.parse().context("invalid --gumbel-c-scale")?;
                if !(args.gumbel_c_scale > 0.0) {
                    bail!("--gumbel-c-scale must be positive");
                }
            }
            "--gumbel-sigma-norm" => {
                args.gumbel_sigma_norm = parse_sigma_norm(&value()?)?;
            }
            "--gumbel-paired-rollouts" => args.gumbel_paired_rollouts = true,
            "--gumbel-ghost-opponents" => args.gumbel_ghost_opponents = true,
            "--gumbel-q-bias-correction" => args.gumbel_q_bias_correction = true,
            "--gumbel-lcb-c" => {
                args.gumbel_lcb_c = value()?.parse().context("invalid --gumbel-lcb-c")?;
                if !(args.gumbel_lcb_c >= 0.0) {
                    bail!("--gumbel-lcb-c must be non-negative");
                }
            }
            "--gumbel-refresh-sample-divisor" => {
                args.gumbel_refresh_sample_divisor = value()?
                    .parse()
                    .context("invalid --gumbel-refresh-sample-divisor")?;
                if args.gumbel_refresh_sample_divisor == 0 {
                    bail!("--gumbel-refresh-sample-divisor must be positive");
                }
            }
            "--gumbel-tta" => {
                args.gumbel_tta = value()?.parse().context("invalid --gumbel-tta")?;
                if args.gumbel_tta < 1 || args.gumbel_tta > 6 {
                    bail!("--gumbel-tta must be in 1..=6");
                }
            }
            "--gumbel-determinizations" => {
                args.gumbel_determinizations = value()?
                    .parse()
                    .context("invalid --gumbel-determinizations")?
            }
            "--gumbel-market-decision-samples" => {
                args.gumbel_market_decision_samples = value()?
                    .parse()
                    .context("invalid --gumbel-market-decision-samples")?;
                if args.gumbel_market_decision_samples == 0 {
                    bail!("--gumbel-market-decision-samples must be positive");
                }
            }
            "--gumbel-exact-endgame-turns" => {
                args.gumbel_exact_endgame_turns = value()?
                    .parse()
                    .context("invalid --gumbel-exact-endgame-turns")?;
                if args.gumbel_exact_endgame_turns > 1 {
                    bail!("--gumbel-exact-endgame-turns currently supports only 0 or 1");
                }
            }
            "--gumbel-blend-weight" => {
                args.gumbel_blend_weight =
                    value()?.parse().context("invalid --gumbel-blend-weight")?
            }
            "--gumbel-parallel-leaf-rollouts" => args.gumbel_parallel_leaf_rollouts = true,
            "--gumbel-exploration" => {
                args.gumbel_exploration = match value()?.as_str() {
                    "on" | "true" | "1" => true,
                    "off" | "false" | "0" => false,
                    other => bail!("invalid --gumbel-exploration {other}; use on|off"),
                }
            }
            "--gumbel-root-menu" => {
                args.gumbel_root_menu = value()?.parse().context("invalid --gumbel-root-menu")?
            }
            "--gumbel-max-root-actions" => {
                args.gumbel_max_root_actions = Some(
                    value()?
                        .parse()
                        .context("invalid --gumbel-max-root-actions")?,
                )
            }
            "--k-interior" => args.k_interior = value()?.parse().context("invalid --k-interior")?,
            "--model-sessions" => {
                args.model_sessions = Some(value()?.parse().context("invalid --model-sessions")?)
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
            "--source-revision" => args.source_revision = Some(value()?),
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
            | Mode::GumbelSuggestServer
            | Mode::GumbelBenchmarkBatch
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
        Mode::GumbelPolicyGame
            | Mode::GumbelSuggestServer
            | Mode::GumbelBenchmarkBatch
            | Mode::GumbelSelfplayTensorCorpus
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
        if args.gumbel_exact_endgame_turns > 0
            && (args.gumbel_table_total || args.gumbel_table_native_q)
        {
            bail!("--gumbel-exact-endgame-turns is incompatible with table-total objectives");
        }
    }
    if args.mode == Mode::GumbelBenchmarkBatch && args.output_dir.is_none() {
        bail!("--gumbel-benchmark-batch requires --output-dir <dir> for per-seed JSONL outputs");
    }
    if args.model_service.is_some() && args.model_manifest.is_none() && !args.allow_model_fallback {
        bail!(
            "real model service use requires --model-manifest unless --allow-model-fallback is set"
        );
    }
    if args.mode == Mode::GumbelSelfplayTensorCorpus
        && args
            .source_revision
            .as_deref()
            .map_or(true, |revision| revision.trim().is_empty())
    {
        bail!("--gumbel-selfplay-tensor-corpus requires non-empty --source-revision");
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
  --source-revision <git>  Exact source revision embedded in generated
                           Gumbel self-play tensor metadata and manifest.
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
  --gumbel-benchmark-batch Play many seeds' complete Gumbel policy games in
                           one process (dynamic worker queue, optional shared
                           bridge; no fixed-chunk long tail).
                           Per seed the search/record behavior is identical to
                           --gumbel-policy-game; writes one JSONL file per seed
                           (gumbel_game_seed_<seed>.jsonl) into --output-dir.
  --output-dir <dir>       Per-seed JSONL output directory for
                           --gumbel-benchmark-batch.
  --gumbel-selfplay-tensor-corpus
                           All-seat Gumbel self-play; every visited root is
                           exported as a v3 expert tensor record with
                           completed-Q targets, improved policy, explicit
                           exact-endgame flags, and real final outcomes.
  --gumbel-n-simulations <n>
                           Total simulation budget per decision [64].
  --gumbel-top-m <n>       Gumbel top-m root candidates [16].
  --gumbel-depth-rounds <n>
                           Root-seat re-entries before leaf valuation [1].
  --gumbel-peek                ORACLE ONLY: search on the true hidden state
                               (leaks hidden info; ceiling measurement, never
                               honest gates or labels)
  --gumbel-table-total         Value simulations by the table total (sum of
                               all seats) instead of the root seat's own
                               score. Gate-aligned cooperative objective.
  --gumbel-leaf-softmix <tau>  Leaf bootstrap = softmax(q/tau)-weighted mean
                               instead of max-Q (bias/variance reduction).
  --gumbel-table-native-q      The model's q head predicts table-scale
                               score-to-go (table-total-trained cycle):
                               table terminals/rollouts, no value shift.
  --gumbel-tta <k>             Symmetry test-time augmentation: average
                               model evals over k rotated board frames
                               (1 = off, max 6). k× eval cost.
  --gumbel-c-visit <c>         Sigma visit constant: sigma = (c_visit +
                               max_visits) * c_scale * norm(q) [50].
  --gumbel-c-scale <c>         Sigma scale; the Gumbel paper shrinks this
                               under noisy Q (0.1 for Atari) [1.0].
  --gumbel-sigma-norm <scheme> Q normalization inside sigma:
                               minmax|zscore|fixed:<scale>|topk:<k> [minmax].
  --gumbel-paired-rollouts     Common-random-number leaf rollouts: share the
                               rollout stream across root actions at equal
                               (world, visit) index so rollout noise cancels
                               in halving comparisons [off].
  --gumbel-ghost-opponents     R1.2A: interior non-root plies advance by the
                               CPU greedy policy (refresh declined) with zero
                               model evals — removes the 3-of-4 opponent
                               eval tax [off].
  --gumbel-q-bias-correction   R0.3: offset unvisited-action model Q by the
                               per-root mean (sim mean - model Q) over
                               visited actions [off].
  --gumbel-lcb-c <c>           R0.4: final action = argmax(mean - c*SE) among
                               actions with >= half the max visit count
                               [0 = last halving survivor].
  --gumbel-refresh-sample-divisor <k>
                               R0.6: refresh hidden-replacement sample
                               searches run at n/k budget [1 = full].
  --table-contention-audit     Replay a decisions ledger (--in) without
                               search; for every decision compare the chosen
                               action vs the best model-Q alternative under
                               the TABLE objective (value-head sums). Bounds
                               the cooperative-play prize (R1.1a).
  --search-stability-probe     Replay a decisions ledger (--in) and re-run
                               root searches repeatedly, unpaired vs paired
                               (CRN) rollouts, equal repeat = equal search
                               seed. Offline kill test for R0.2.
  --probe-stride <k>           Sample every k-th ledger decision [7].
  --probe-repeats <n>          Search repeats per (root, variant) [6].
  --probe-max-roots <n>        Total sampled roots cap [100].
  --puzzle-bank                Replay a decisions ledger (--in) and resolve
                               stride-selected roots with the configured
                               search, worker-pooled across seeds
                               (--model-sessions/--shared-model-session).
                               Mega flags + repeats>=2 = frozen bank;
                               candidate flags + repeats=1 = screen run
                               (scored by analyze_puzzle_screen).
  --gumbel-determinizations <n>
                           Hidden-order determinizations cycled per action [4].
  --gumbel-market-decision-samples <n>
                           Hidden replacement samples used before accepting
                           an optional three-of-a-kind refresh [8].
  --gumbel-exact-endgame-turns <0|1>
                           Replace model/search with exact own final-score
                           selection on the last personal turn [0].
  --gumbel-blend-weight <w>
                           Leaf value = w*model bootstrap + (1-w)*greedy
                           rollout [0.5].
  --gumbel-parallel-leaf-rollouts
                           Resolve independent terminal greedy rollouts on the
                           Rayon pool; deterministic opt-in execution ablation.
  --gumbel-exploration <on|off>
                           Gumbel exploration noise at the root [off; selfplay
                           mode defaults on].
  --gumbel-max-root-actions <n>
                           Optional model-prior-ranked cap on root candidates.
  --gumbel-root-menu <n>   Root menu enumeration cap before model ranking;
                           0 keeps the full legal set [256].
  --k-interior <n>         Interior-ply menu cap inside simulations [16].
  --model-sessions <n>     Cap on persistent model-backed seed workers (with
                           --shared-model-session this is the parallel game
                           count, not the bridge-process count).
  --shared-model-session   One bridge process (one CUDA context) serving all
                           workers with cross-worker request batching.
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
    let (prelude, staged) = greedy_market_choice(game, None)?;
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

    let expected_prelude: MarketPrelude = serde_json::from_value(
        root_replay
            .get("market_prelude")
            .cloned()
            .context("root_replay.market_prelude missing")?,
    )?;
    if !game
        .free_three_of_a_kind_choices()?
        .contains(&expected_prelude)
    {
        bail!("recorded market prelude is not legal during reconstruction");
    }
    let staged = game.preview_market_prelude(&expected_prelude)?;
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
            let (prelude, staged) = greedy_market_choice(&game, None)?;
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
    let score_means = terminal_scores.iter().map(score_mean).collect::<Vec<_>>();
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
            exact_afterstate_score_decomposition_active: None,
            active_seat: None,
            final_score_vector: &shard.final_score_vector,
            rank_vector: &shard.rank_vector,
            score_decomposition: &shard.score_decomposition,
            improved_policy: None,
            search_root_value: None,
            exact_endgame: None,
            record_count: shard.record_count,
            compression: args.tensor_compression,
        },
    )?;
    let checksum = sha256_file_hex(&args.out)?;
    write_expert_tensor_manifest(&args.manifest, args, &shard, &checksum, &metadata)?;
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
            exact_afterstate_score_decomposition_active: None,
            active_seat: None,
            final_score_vector: &shard.final_score_vector,
            rank_vector: &shard.rank_vector,
            score_decomposition: &shard.score_decomposition,
            improved_policy: None,
            search_root_value: None,
            exact_endgame: None,
            record_count: shard.record_count,
            compression: args.tensor_compression,
        },
    )?;
    let checksum = sha256_file_hex(&args.out)?;
    write_expert_tensor_manifest(&args.manifest, args, &shard, &checksum, &metadata)?;
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
            exact_afterstate_score_decomposition_active: None,
            active_seat: None,
            final_score_vector: &shard.final_score_vector,
            rank_vector: &shard.rank_vector,
            score_decomposition: &shard.score_decomposition,
            improved_policy: None,
            search_root_value: None,
            exact_endgame: None,
            record_count: shard.record_count,
            compression: args.tensor_compression,
        },
    )?;
    let checksum = sha256_file_hex(&args.out)?;
    write_expert_tensor_manifest(&args.manifest, args, &shard, &checksum, &metadata)?;
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
    // Static Rayon used 2x as many chunks as threads, but could execute only
    // `current_num_threads()` chunks concurrently. Preserve that actual
    // session/concurrency ceiling now that every worker is an OS thread.
    let target_workers = rayon::current_num_threads().max(1);
    let (mut per_seed, _) = run_dynamic_seed_workers(
        &seeds,
        target_workers,
        |_| model_state_worker_session(args),
        |model_session, seed_u64| {
            let records = export_model_state_search_bootstrap_seed_records_with_session(
                args,
                seed_u64,
                model_session,
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
            let records_done = completed_records
                .fetch_add(shard.record_count as u64, Ordering::Relaxed)
                + shard.record_count as u64;
            log_seed_export_progress(
                "model-state search-bootstrap tensor",
                done,
                total_seeds,
                records_done,
                started,
            );
            Ok(shard)
        },
    )?;
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
            exact_afterstate_score_decomposition_active: None,
            active_seat: None,
            final_score_vector: &shard.final_score_vector,
            rank_vector: &shard.rank_vector,
            score_decomposition: &shard.score_decomposition,
            improved_policy: None,
            search_root_value: None,
            exact_endgame: None,
            record_count: shard.record_count,
            compression: args.tensor_compression,
        },
    )?;
    let checksum = sha256_file_hex(&args.out)?;
    write_expert_tensor_manifest(&args.manifest, args, &shard, &checksum, &metadata)?;
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

/// Runs variable-duration seeds on a fixed number of persistent workers.
///
/// Each worker owns its expensive model session and cache for its full
/// lifetime, while the atomic queue assigns the next unstarted seed whenever
/// that worker becomes free. This preserves the hard concurrency/session cap
/// without the long-tail collapse caused by static contiguous seed chunks.
/// Results are returned with their seed because completion order is expected
/// to differ from input order.
fn run_dynamic_seed_workers<Worker, Output, InitializeWorker, RunSeed>(
    seeds: &[u64],
    max_workers: usize,
    initialize_worker: InitializeWorker,
    run_seed: RunSeed,
) -> Result<(Vec<(u64, Output)>, Vec<Worker>)>
where
    Worker: Send,
    Output: Send,
    InitializeWorker: Fn(usize) -> Result<Worker> + Sync,
    RunSeed: Fn(&mut Worker, u64) -> Result<Output> + Sync,
{
    if seeds.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }
    let worker_count = max_workers.max(1).min(seeds.len());
    let next_seed_index = AtomicUsize::new(0);
    let failed = AtomicBool::new(false);
    let joined = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(worker_count);
        for worker_index in 0..worker_count {
            let next_seed_index = &next_seed_index;
            let failed = &failed;
            let initialize_worker = &initialize_worker;
            let run_seed = &run_seed;
            handles.push(
                scope.spawn(move || -> Result<(Worker, Vec<(u64, Output)>)> {
                    let mut worker = match initialize_worker(worker_index)
                        .with_context(|| format!("initializing dynamic seed worker {worker_index}"))
                    {
                        Ok(worker) => worker,
                        Err(error) => {
                            failed.store(true, Ordering::Release);
                            return Err(error);
                        }
                    };
                    let mut outputs = Vec::new();
                    loop {
                        if failed.load(Ordering::Acquire) {
                            break;
                        }
                        let seed_index = next_seed_index.fetch_add(1, Ordering::Relaxed);
                        let Some(seed_u64) = seeds.get(seed_index).copied() else {
                            break;
                        };
                        match run_seed(&mut worker, seed_u64) {
                            Ok(output) => outputs.push((seed_u64, output)),
                            Err(error) => {
                                failed.store(true, Ordering::Release);
                                return Err(error);
                            }
                        }
                    }
                    Ok((worker, outputs))
                }),
            );
        }
        handles
            .into_iter()
            .map(std::thread::ScopedJoinHandle::join)
            .collect::<Vec<_>>()
    });

    let mut outputs = Vec::with_capacity(seeds.len());
    let mut workers = Vec::with_capacity(worker_count);
    for worker_result in joined {
        let (worker, mut worker_outputs) =
            worker_result.map_err(|_| anyhow::anyhow!("dynamic seed worker panicked"))??;
        workers.push(worker);
        outputs.append(&mut worker_outputs);
    }
    if outputs.len() != seeds.len() {
        bail!(
            "dynamic seed scheduler completed {} of {} seeds without an explicit error",
            outputs.len(),
            seeds.len()
        );
    }
    Ok((outputs, workers))
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
        Some(command) => {
            match ModelServiceSession::spawn(command, &BridgeConfig::from_args(args)) {
                Ok(session) => Ok(Some(session)),
                Err(error) if args.allow_model_fallback => {
                    eprintln!(
                        "model service unavailable for worker; using fallback priors: {error}"
                    );
                    Ok(None)
                }
                Err(error) => Err(error),
            }
        }
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
        market_decision_samples: args.gumbel_market_decision_samples,
        exact_endgame_turns: args.gumbel_exact_endgame_turns,
        rollout_blend_weight: args.gumbel_blend_weight,
        parallel_leaf_rollouts: args.gumbel_parallel_leaf_rollouts,
        rollout_max_actions: args.max_actions,
        rollout_top_k: args.rollout_top_k,
        k_interior: args.k_interior,
        exploration: args.gumbel_exploration,
        peek_true_hidden: args.gumbel_peek,
        table_total: args.gumbel_table_total,
        table_native_q: args.gumbel_table_native_q,
        leaf_softmix_temp: args.gumbel_leaf_softmix,
        c_visit: args.gumbel_c_visit,
        c_scale: args.gumbel_c_scale,
        sigma_normalization: args.gumbel_sigma_norm,
        paired_rollouts: args.gumbel_paired_rollouts,
        ghost_opponents: args.gumbel_ghost_opponents,
        q_bias_correction: args.gumbel_q_bias_correction,
        lcb_c: args.gumbel_lcb_c,
        refresh_sample_divisor: args.gumbel_refresh_sample_divisor,
        search_seed,
        ..gumbel::GumbelConfig::default()
    }
}

/// Parses `--gumbel-sigma-norm`: `minmax` | `zscore` | `fixed:<scale>` |
/// `topk:<k>`.
fn parse_sigma_norm(raw: &str) -> Result<gumbel::SigmaNormalization> {
    match raw {
        "minmax" => return Ok(gumbel::SigmaNormalization::MinMax),
        "zscore" => return Ok(gumbel::SigmaNormalization::ZScore),
        _ => {}
    }
    if let Some(scale) = raw.strip_prefix("fixed:") {
        let scale: f64 = scale
            .parse()
            .context("invalid --gumbel-sigma-norm fixed:<scale>")?;
        if !(scale > 0.0) {
            bail!("--gumbel-sigma-norm fixed:<scale> requires a positive scale");
        }
        return Ok(gumbel::SigmaNormalization::FixedScale(scale));
    }
    if let Some(k) = raw.strip_prefix("topk:") {
        let k: usize = k.parse().context("invalid --gumbel-sigma-norm topk:<k>")?;
        if k < 2 {
            bail!("--gumbel-sigma-norm topk:<k> requires k >= 2");
        }
        return Ok(gumbel::SigmaNormalization::TopKRange(k));
    }
    bail!("invalid --gumbel-sigma-norm {raw}; use minmax|zscore|fixed:<scale>|topk:<k>")
}

fn gumbel_search_seed(seed_u64: u64, ply_index: usize) -> u64 {
    gumbel::splitmix64(seed_u64 ^ gumbel::splitmix64(ply_index as u64 ^ 0x6706_2026))
}

fn eval_request_for_row(row: &gumbel::EvalRow, packed: bool) -> Result<Value> {
    let public_hash = row.staged.public_state().canonical_hash();
    eval_request_for_row_with_public_hash(row, packed, &public_hash)
}

/// `eval_request_for_row` with the (already computed) public-state hash
/// passed in, so the dedup path hashes each row's public state exactly once.
fn eval_request_for_row_with_public_hash(
    row: &gumbel::EvalRow,
    packed: bool,
    public_hash: &blake3::Hash,
) -> Result<Value> {
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
    let request = json!({
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
    });
    if packed {
        return pack_eval_request(request);
    }
    Ok(request)
}

/// Converts a raw eval request into its packed-features form. This is a pure
/// function of the raw request, so a raw request fully determines its packed
/// twin (which lets the dedup path key on cheap raw payloads and pack only
/// the unique rows it actually sends).
fn pack_eval_request(mut request: Value) -> Result<Value> {
    let packed_features = packed_features_for_request(&request)?;
    let object = request
        .as_object_mut()
        .expect("eval request is always an object");
    // The bridge builds tensors straight from the packed arrays; the raw
    // token/action dictionaries would only duplicate megabytes per root.
    object.remove("legal_actions");
    object.remove("public_tokens");
    object.insert("packed_features".to_owned(), packed_features);
    Ok(request)
}

/// Precomputed model-input features for an eval request, base64-encoded
/// little-endian arrays. Row-major shapes: tokens `T x 41` f32, actions
/// `A x 61` f32, relation tail `A x (T + A)` u8 (token columns first).
fn packed_features_for_request(request: &Value) -> Result<Value> {
    use base64::Engine as _;

    let token_rows = feature_tensors::public_token_features(request)?;
    let action_rows = feature_tensors::semantic_public_token_action_features(request)?;
    let token_count = token_rows.len();
    let action_count = action_rows.len();
    let relation_tail = feature_tensors::action_relation_tail(request, token_count, action_count)?;

    let mut token_bytes = Vec::with_capacity(token_count * PUBLIC_TOKEN_FEATURE_DIM * 4);
    for value in token_rows.iter().flatten() {
        token_bytes.extend_from_slice(&value.to_le_bytes());
    }
    let mut action_bytes =
        Vec::with_capacity(action_count * SEMANTIC_PUBLIC_TOKEN_ACTION_FEATURE_DIM * 4);
    for value in action_rows.iter().flatten() {
        action_bytes.extend_from_slice(&value.to_le_bytes());
    }
    let engine = base64::engine::general_purpose::STANDARD;
    Ok(json!({
        "token_count": token_count,
        "action_count": action_count,
        "token_feature_dim": PUBLIC_TOKEN_FEATURE_DIM,
        "action_feature_dim": SEMANTIC_PUBLIC_TOKEN_ACTION_FEATURE_DIM,
        "tokens_f32_b64": engine.encode(&token_bytes),
        "actions_f32_b64": engine.encode(&action_bytes),
        "relation_tail_u8_b64": engine.encode(&relation_tail),
    }))
}

/// Per-worker bridge handle: either an owned session (one CUDA context per
/// worker) or a client of the process-wide shared aggregator bridge (one
/// CUDA context total, cross-worker batching).
enum ChunkBridge {
    Owned(Option<ModelServiceSession>),
    Shared(SharedBridgeClient),
}

impl ChunkBridge {
    fn supports_packed_features(&self) -> bool {
        match self {
            ChunkBridge::Owned(Some(session)) => session.supports_packed_features(),
            ChunkBridge::Owned(None) => false,
            ChunkBridge::Shared(client) => client.supports_packed_features(),
        }
    }
}

/// Rows-saved accounting for the dedup + cache eval path.
#[derive(Debug, Default, Clone, Copy)]
struct EvalDedupStats {
    /// Rows the search asked to evaluate.
    rows_requested: u64,
    /// Unique rows actually sent to the model bridge.
    rows_sent: u64,
    /// Rows served from the cross-call cache.
    cache_hits: u64,
}

impl EvalDedupStats {
    fn accumulate_into(
        &self,
        rows_requested: &AtomicU64,
        rows_sent: &AtomicU64,
        cache_hits: &AtomicU64,
    ) {
        rows_requested.fetch_add(self.rows_requested, Ordering::Relaxed);
        rows_sent.fetch_add(self.rows_sent, Ordering::Relaxed);
        cache_hits.fetch_add(self.cache_hits, Ordering::Relaxed);
    }
}

fn log_eval_dedup_summary(label: &str, rows_requested: u64, rows_sent: u64, cache_hits: u64) {
    if rows_requested == 0 {
        return;
    }
    let saved = rows_requested.saturating_sub(rows_sent);
    eprintln!(
        "[real-root-exporter] {label} eval dedup: {rows_requested} rows requested, {rows_sent} sent to bridge ({:.1}% saved), {cache_hits} cache hits",
        100.0 * saved as f64 / rows_requested as f64,
    );
}

/// Cache entries are cleared wholesale past this bound; at k_interior <= 64
/// this stays well under ~60 MB per worker.
const EVAL_ROW_CACHE_MAX_ENTRIES: usize = 50_000;

/// Cross-call eval-output cache keyed by `eval_row_key`, a blake3 hash over
/// the semantic inputs that fully determine the serialized eval request
/// (public state, public supply, prelude cleanup projection, and the action
/// menu with its scores). The request payload fully determines the bridge
/// output, so identical keys are safe to reuse across simulations, plies,
/// and games handled by one worker.
struct EvalRowCache {
    entries: HashMap<[u8; 32], gumbel::EvalOut>,
    stats: EvalDedupStats,
}

/// Persistent state for one dynamically scheduled model-backed game worker.
/// The session/client and eval cache intentionally survive across seeds.
struct GumbelSeedWorker {
    bridge: ChunkBridge,
    eval_cache: EvalRowCache,
}

impl EvalRowCache {
    fn new() -> Self {
        Self {
            entries: HashMap::new(),
            stats: EvalDedupStats::default(),
        }
    }
}

/// Reference dedup key: blake3 over the serialized raw (unpacked) request
/// bytes plus a wire-format marker. The raw payload is an information
/// superset of the packed payload (packing is a pure function of it), so it
/// keys either form. Superseded on the hot path by `eval_row_key`, which
/// hashes the same semantic inputs without building the request; kept as the
/// ground-truth definition of request equality for tests.
#[cfg(test)]
fn eval_request_key(raw_request: &Value, packed: bool) -> Result<[u8; 32]> {
    let bytes =
        serde_json::to_vec(raw_request).context("serializing eval request for dedup key")?;
    let mut hasher = blake3::Hasher::new();
    hasher.update(&[u8::from(packed)]);
    hasher.update(&bytes);
    Ok(*hasher.finalize().as_bytes())
}

/// Cheap dedup key over exactly the semantic inputs `eval_request_for_row`
/// consumes, so cache-hit rows never pay full request construction. Returns
/// the key plus the row's public-state hash (reused as the request's
/// `state_hash` when the row turns out to be unique).
///
/// Completeness argument (equal key <=> byte-identical raw request):
/// - `public_state().canonical_hash()` covers config (incl. scoring cards),
///   boards, market, current player, and completed turns — everything
///   `public_tokens`, `score_game`, `market_components`, and the request's
///   own `state_hash`/`active_seat` fields read from the staged state.
/// - `public_supply()` is NOT determined by the public state (discards are
///   outside `PublicGameState`) and feeds the supply token, so it is hashed
///   separately. It is invariant under `redeterminize_hidden`, keeping dedup
///   across determinizations intact.
/// - The prelude enters the request only through per-action
///   `cleanup_choice()`, so exactly that projection (three-of-a-kind flag +
///   wipe slot lists) is hashed, and only when the menu is non-empty.
/// - Per afterstate: the canonical action serialization (the request embeds
///   its sha256 as `action_id`, and every other action-derived token field
///   is a function of these bytes), the immediate base score, and the exact
///   afterstate score. Non-finite scores all collapse to JSON `null`, so
///   they hash as one marker; finite scores hash by bit pattern, matching
///   JSON f64 formatting equality.
/// The leading domain byte separates this scheme from `eval_request_key`,
/// and every variable-length field is length-prefixed to keep the framing
/// injective.
fn eval_row_key(row: &gumbel::EvalRow, packed: bool) -> Result<([u8; 32], blake3::Hash)> {
    let staged = &row.staged;
    let public_hash = staged.public_state().canonical_hash();
    let supply = staged.public_supply();
    let mut hasher = blake3::Hasher::new();
    hasher.update(&[0xE1, u8::from(packed)]);
    hasher.update(public_hash.as_bytes());
    hasher.update(&supply.wildlife_bag);
    hasher.update(&supply.unseen_tile_terrain_capacity);
    hasher.update(&supply.unseen_tile_wildlife_capacity);
    hasher.update(&supply.unseen_keystones_by_terrain);
    hasher.update(&supply.unseen_dual_terrain_pairs);
    hasher.update(&(row.afterstates.len() as u64).to_le_bytes());
    if !row.afterstates.is_empty() {
        hasher.update(&[u8::from(row.prelude.replace_three_of_a_kind)]);
        hasher.update(&(row.prelude.wildlife_wipes.len() as u64).to_le_bytes());
        for wipe in &row.prelude.wildlife_wipes {
            hasher.update(&(wipe.slots.len() as u64).to_le_bytes());
            for slot in &wipe.slots {
                hasher.update(&[slot.index() as u8]);
            }
        }
        for afterstate in &row.afterstates {
            let action_bytes = serde_json::to_vec(&afterstate.candidate.action)
                .context("serializing action for dedup key")?;
            hasher.update(&(action_bytes.len() as u64).to_le_bytes());
            hasher.update(&action_bytes);
            hasher.update(&afterstate.candidate.resulting_base_score.to_le_bytes());
            let exact = afterstate.exact_score_active;
            if exact.is_finite() {
                hasher.update(&[1]);
                hasher.update(&exact.to_bits().to_le_bytes());
            } else {
                // serde_json renders every non-finite float as `null`.
                hasher.update(&[0]);
            }
        }
    }
    Ok((*hasher.finalize().as_bytes(), public_hash))
}

/// Converts one bridge eval into search outputs for its row.
fn eval_out_for_row(row: &gumbel::EvalRow, eval: &ModelEval) -> Result<gumbel::EvalOut> {
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
        value_vector: eval.value.clone(),
    })
}

/// Evaluates rows through `eval_unique` after deduplicating identical
/// requests within the batch and serving repeats from `cache`. Results are
/// fanned back out in the original row order. Rows with identical request
/// payloads have identical menus (action ids and exact afterstate scores are
/// part of the payload), so sharing one bridge eval across them is exact.
fn evaluate_rows_deduped<F>(
    rows: &[gumbel::EvalRow],
    packed: bool,
    cache: &mut EvalRowCache,
    mut eval_unique: F,
) -> Result<Vec<gumbel::EvalOut>>
where
    F: FnMut(&[Value], &[&gumbel::EvalRow]) -> Result<Vec<ModelEval>>,
{
    cache.stats.rows_requested += rows.len() as u64;
    let mut outputs: Vec<Option<gumbel::EvalOut>> = Vec::with_capacity(rows.len());
    // Per row: index into the unique batch, or None when cache-served.
    let mut row_slots: Vec<Option<usize>> = Vec::with_capacity(rows.len());
    let mut unique_requests: Vec<Value> = Vec::new();
    let mut unique_rows: Vec<&gumbel::EvalRow> = Vec::new();
    let mut unique_keys: Vec<[u8; 32]> = Vec::new();
    let mut key_to_unique: HashMap<[u8; 32], usize> = HashMap::new();

    for row in rows {
        // Key on a cheap structural hash of the request's semantic inputs;
        // duplicate and cache-served rows never pay request construction or
        // feature packing.
        let (key, public_hash) = eval_row_key(row, packed)?;
        if let Some(cached) = cache.entries.get(&key) {
            cache.stats.cache_hits += 1;
            outputs.push(Some(cached.clone()));
            row_slots.push(None);
            continue;
        }
        let unique_index = match key_to_unique.entry(key) {
            std::collections::hash_map::Entry::Occupied(entry) => *entry.get(),
            std::collections::hash_map::Entry::Vacant(entry) => {
                let request = eval_request_for_row_with_public_hash(row, packed, &public_hash)?;
                unique_requests.push(request);
                unique_rows.push(row);
                unique_keys.push(key);
                *entry.insert(unique_requests.len() - 1)
            }
        };
        outputs.push(None);
        row_slots.push(Some(unique_index));
    }

    if !unique_requests.is_empty() {
        cache.stats.rows_sent += unique_requests.len() as u64;
        let evals = eval_unique(&unique_requests, &unique_rows)?;
        if evals.len() != unique_requests.len() {
            bail!(
                "model bridge returned {} results for {} unique rows",
                evals.len(),
                unique_requests.len()
            );
        }
        let unique_outs = unique_rows
            .iter()
            .zip(evals.iter())
            .map(|(row, eval)| eval_out_for_row(row, eval))
            .collect::<Result<Vec<_>>>()?;
        for (key, out) in unique_keys.iter().zip(unique_outs.iter()) {
            if cache.entries.len() >= EVAL_ROW_CACHE_MAX_ENTRIES {
                cache.entries.clear();
            }
            cache.entries.insert(*key, out.clone());
        }
        for (slot, output) in row_slots.iter().zip(outputs.iter_mut()) {
            if let Some(unique_index) = slot {
                *output = Some(unique_outs[*unique_index].clone());
            }
        }
    }

    outputs
        .into_iter()
        .map(|output| output.context("deduped eval left a row without output"))
        .collect()
}

/// The rotation-transformed twin of an eval row: every board rotated,
/// every candidate action expressed in the rotated frame. Exact afterstate
/// scores are rotation-invariant and copied verbatim, so the transformed
/// row's derived Q differs from the original only through the model's
/// score-to-go/prior outputs — which is exactly the decorrelation TTA wants.
fn rotated_eval_row(row: &gumbel::EvalRow, steps: u8) -> gumbel::EvalRow {
    gumbel::EvalRow {
        staged: row.staged.with_rotated_boards(steps),
        prelude: row.prelude.clone(),
        afterstates: row
            .afterstates
            .iter()
            .map(|afterstate| CandidateAfterstate {
                candidate: GreedyCandidate {
                    action: afterstate.candidate.action.rotated(steps),
                    resulting_base_score: afterstate.candidate.resulting_base_score,
                    immediate_rank: afterstate.candidate.immediate_rank,
                },
                state: afterstate.state.with_rotated_boards(steps),
                exact_score_active: afterstate.exact_score_active,
                exact_score_decomposition_active: afterstate.exact_score_decomposition_active,
                apply_truncated: afterstate.apply_truncated,
            })
            .collect(),
    }
}

/// Sends one request batch through the chunk bridge with the standard
/// uniform-prior fallback semantics. `action_counts` sizes the fallback
/// evals (one per request, aligned).
fn chunk_bridge_eval_batch(
    bridge: &mut ChunkBridge,
    allow_model_fallback: bool,
    requests: &[Value],
    action_counts: &[usize],
) -> Result<Vec<ModelEval>> {
    match bridge {
        ChunkBridge::Shared(client) => client.eval_batch(requests),
        ChunkBridge::Owned(session_slot) => {
            if let Some(session) = session_slot.as_mut() {
                match session.eval_batch(requests) {
                    Ok(evals) => Ok(evals),
                    Err(error) if allow_model_fallback => {
                        eprintln!(
                            "model service batch eval failed; falling back to uniform priors: {error}"
                        );
                        *session_slot = None;
                        Ok(action_counts
                            .iter()
                            .map(|count| uniform_model_eval(*count))
                            .collect())
                    }
                    Err(error) => Err(error),
                }
            } else if allow_model_fallback {
                Ok(action_counts
                    .iter()
                    .map(|count| uniform_model_eval(*count))
                    .collect())
            } else {
                bail!("gumbel search requires a model service or --allow-model-fallback");
            }
        }
    }
}

/// Batched leaf evaluator over the JSONL model bridge. Identical rows are
/// deduplicated within each batch and served from a cross-call cache; only
/// unique unseen rows pay a bridge round-trip. Falls back to
/// exact-afterstate-only values (uniform priors) when the bridge is
/// unavailable and fallback is allowed. With `tta_rotations > 1`, each
/// unique row is additionally evaluated on rotated board frames and the
/// model outputs are averaged (symmetry test-time augmentation); the cache
/// stores the averaged result.
struct BridgeLeafEvaluator<'a> {
    bridge: &'a mut ChunkBridge,
    allow_model_fallback: bool,
    cache: &'a mut EvalRowCache,
    tta_rotations: u8,
}

impl gumbel::LeafEvaluator for BridgeLeafEvaluator<'_> {
    fn evaluate_batch(&mut self, rows: &[gumbel::EvalRow]) -> Result<Vec<gumbel::EvalOut>> {
        let packed = self.bridge.supports_packed_features();
        let bridge = &mut *self.bridge;
        let allow_model_fallback = self.allow_model_fallback;
        let tta_rotations = self.tta_rotations.clamp(1, 6);
        evaluate_rows_deduped(rows, packed, self.cache, |requests, unique_rows| {
            let action_counts: Vec<usize> = unique_rows
                .iter()
                .map(|row| row.afterstates.len())
                .collect();
            let base =
                chunk_bridge_eval_batch(bridge, allow_model_fallback, requests, &action_counts)?;
            if tta_rotations <= 1 {
                return Ok(base);
            }
            let mut variants: Vec<Vec<ModelEval>> = vec![base];
            for steps in 1..tta_rotations {
                let rotated_rows: Vec<gumbel::EvalRow> = unique_rows
                    .iter()
                    .map(|row| rotated_eval_row(row, steps))
                    .collect();
                let rotated_requests = rotated_rows
                    .iter()
                    .map(|row| eval_request_for_row(row, packed))
                    .collect::<Result<Vec<_>>>()?;
                variants.push(chunk_bridge_eval_batch(
                    bridge,
                    allow_model_fallback,
                    &rotated_requests,
                    &action_counts,
                )?);
            }
            (0..unique_rows.len())
                .map(|row_index| {
                    let per_row: Vec<&ModelEval> =
                        variants.iter().map(|batch| &batch[row_index]).collect();
                    average_model_evals(&per_row)
                })
                .collect()
        })
    }
}

fn score_means_from_breakdowns(scores: &[ScoreBreakdown]) -> Vec<ScoreMean> {
    scores.iter().map(score_mean).collect()
}

fn gumbel_search_metadata(args: &Args, result: &gumbel::GumbelSearchResult) -> Value {
    json!({
        "n_simulations": args.gumbel_n_simulations,
        "top_m": args.gumbel_top_m,
        "depth_rounds": args.gumbel_depth_rounds,
        "determinization_samples": args.gumbel_determinizations,
        "market_decision_samples": args.gumbel_market_decision_samples,
        "exact_endgame_turns": args.gumbel_exact_endgame_turns,
        "rollout_blend_weight": args.gumbel_blend_weight,
        "parallel_leaf_rollouts": args.gumbel_parallel_leaf_rollouts,
        "exploration": args.gumbel_exploration,
        "k_interior": args.k_interior,
        "max_root_actions": args.gumbel_max_root_actions,
        "root_menu": args.gumbel_root_menu,
        "table_total": args.gumbel_table_total,
        "table_native_q": args.gumbel_table_native_q,
        "leaf_softmix": args.gumbel_leaf_softmix,
        "tta_rotations": args.gumbel_tta,
        "simulations_run": result.simulations_run,
    })
}

fn gumbel_selfplay_root_record(
    row: &gumbel::EvalRow,
    result: &gumbel::GumbelSearchResult,
    exact_endgame: bool,
    market_branches_searched: usize,
    market_chance_samples: usize,
    total_simulations_run: usize,
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
    let exact_score_decomposition: Vec<Value> = row
        .afterstates
        .iter()
        .map(|afterstate| json!(afterstate.exact_score_decomposition_active))
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
        "ruleset_id": RULESET_ID,
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
        "exact_afterstate_score_decomposition_active": exact_score_decomposition,
        "per_action_Q_valid": result
            .visit_counts
            .iter()
            .map(|visits| json!(*visits > 0))
            .collect::<Vec<_>>(),
        "selected_action": selected_action_id,
        "improved_policy": result.improved_policy,
        "search_root_value": result.root_value,
        "exact_endgame": exact_endgame,
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
            "free_three_of_a_kind_choice": if row.prelude.replace_three_of_a_kind {
                "accept"
            } else if row.staged.market().three_of_a_kind().is_some() {
                "decline"
            } else {
                "not_available"
            },
            "market_branches_searched": market_branches_searched,
            "market_chance_samples": market_chance_samples,
            "total_simulations_run": total_simulations_run,
            "exact_endgame": exact_endgame,
            "search": gumbel_search_metadata(args, result),
        },
    }))
}

fn backfill_final_outcome(records: &mut [Value], final_scores: &[ScoreBreakdown]) -> Result<()> {
    let means = score_means_from_breakdowns(final_scores);
    let final_score_vector: Vec<Value> = means.iter().map(|mean| json!(mean.total)).collect();
    let decomposition = score_decomposition(&means);
    let ranks: Vec<Value> = rank_vector(&means)
        .into_iter()
        .map(|rank| json!(rank))
        .collect();
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
    eval_cache: &mut EvalRowCache,
) -> Result<Vec<Value>> {
    let config = GameConfig::research_aaaaa(args.player_count)?;
    let mut game = GameState::new(config, GameSeed::from_u64(seed_u64))
        .with_context(|| format!("creating gumbel selfplay seed {seed_u64}"))?;
    let mut records = Vec::new();
    let mut ply_index = 0usize;
    while !game.is_game_over() && ply_index < args.plies_per_seed {
        let cfg = gumbel_config_from_args(args, gumbel_search_seed(seed_u64, ply_index));
        let mut evaluator = BridgeLeafEvaluator {
            bridge,
            allow_model_fallback: args.allow_model_fallback,
            cache: eval_cache,
            tta_rotations: args.gumbel_tta,
        };
        let Some(decision) = gumbel::gumbel_search_for_state(
            &game,
            gumbel_root_menu_limit(args),
            &mut evaluator,
            &cfg,
        )?
        else {
            break;
        };
        let market_branches_searched = decision.market_branches_searched;
        let market_chance_samples = decision.market_chance_samples;
        let total_simulations_run = decision.total_simulations_run;
        let exact_endgame = decision.exact_endgame;
        let row = decision.row;
        let result = decision.result;
        let record = gumbel_selfplay_root_record(
            &row,
            &result,
            exact_endgame,
            market_branches_searched,
            market_chance_samples,
            total_simulations_run,
            seed_u64,
            ply_index,
            args,
        )?;
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
    if args
        .source_revision
        .as_deref()
        .map_or(true, |revision| revision.trim().is_empty())
    {
        bail!("gumbel selfplay tensor export requires non-empty source_revision");
    }
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
    let eval_rows_requested = AtomicU64::new(0);
    let eval_rows_sent = AtomicU64::new(0);
    let eval_cache_hits = AtomicU64::new(0);
    let seeds = (args.first_seed..seed_end).collect::<Vec<_>>();
    let target_workers = args
        .model_sessions
        .unwrap_or_else(|| rayon::current_num_threads().max(1))
        .max(1);
    // Shared bridge: one CUDA context serving all workers with cross-worker
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
            model_bridge::shared_row_cap(),
        )?)
    } else {
        None
    };
    let (mut per_seed, workers) = run_dynamic_seed_workers(
        &seeds,
        target_workers,
        |_| {
            let bridge = match &shared_bridge {
                Some(shared) => ChunkBridge::Shared(shared.client()),
                None => ChunkBridge::Owned(model_state_worker_session(args)?),
            };
            Ok(GumbelSeedWorker {
                bridge,
                eval_cache: EvalRowCache::new(),
            })
        },
        |worker, seed_u64| {
            let records = play_gumbel_selfplay_seed(
                args,
                seed_u64,
                &mut worker.bridge,
                &mut worker.eval_cache,
            )
            .with_context(|| format!("exporting gumbel selfplay seed {seed_u64}"))?;
            let shard = ExpertTensorShardData::from_records(&records).with_context(|| {
                format!("extracting gumbel selfplay tensor features for seed {seed_u64}")
            })?;
            let done = completed_seeds.fetch_add(1, Ordering::Relaxed) + 1;
            let records_done = completed_records
                .fetch_add(shard.record_count as u64, Ordering::Relaxed)
                + shard.record_count as u64;
            log_seed_export_progress(
                "gumbel selfplay tensor",
                done,
                total_seeds,
                records_done,
                started,
            );
            Ok(shard)
        },
    )?;
    for worker in &workers {
        worker.eval_cache.stats.accumulate_into(
            &eval_rows_requested,
            &eval_rows_sent,
            &eval_cache_hits,
        );
    }
    log_eval_dedup_summary(
        "gumbel selfplay tensor",
        eval_rows_requested.load(Ordering::Relaxed),
        eval_rows_sent.load(Ordering::Relaxed),
        eval_cache_hits.load(Ordering::Relaxed),
    );
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
    if shard.exact_endgame_field_records != shard.record_count {
        bail!(
            "gumbel selfplay shard has {} explicit exact-endgame flags for {} records",
            shard.exact_endgame_field_records,
            shard.record_count
        );
    }
    if shard.structured_value_field_records != shard.record_count {
        bail!(
            "gumbel selfplay shard has {} structured-value records for {} records",
            shard.structured_value_field_records,
            shard.record_count
        );
    }
    let mut metadata = expert_tensor_shard_metadata(args, &shard);
    let teacher_model = model_artifact_identity(args)?;
    let generator = generator_artifact_identity()?;
    if let Some(object) = metadata.as_object_mut() {
        object.insert("version".to_owned(), json!(EXPERT_SHARD_VERSION_V4));
        object.insert("schema_id".to_owned(), json!(EXPERT_TENSOR_SCHEMA_ID_V4));
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
                "market_decision_samples": args.gumbel_market_decision_samples,
                "exact_endgame_turns": args.gumbel_exact_endgame_turns,
                "rollout_blend_weight": args.gumbel_blend_weight,
                "parallel_leaf_rollouts": args.gumbel_parallel_leaf_rollouts,
                "exploration": args.gumbel_exploration,
                "peek": args.gumbel_peek,
                "table_total": args.gumbel_table_total,
                "table_native_q": args.gumbel_table_native_q,
                "leaf_softmix": args.gumbel_leaf_softmix,
                "tta": args.gumbel_tta,
                "k_interior": args.k_interior,
                "max_root_actions": args.gumbel_max_root_actions,
                "root_menu": args.gumbel_root_menu,
            }),
        );
        object.insert("teacher_model".to_owned(), teacher_model);
        object.insert("generator".to_owned(), generator);
        object.insert(
            "execution".to_owned(),
            json!({
                "rayon_threads_requested": args.rayon_threads,
                "rayon_current_num_threads": rayon::current_num_threads(),
                "model_sessions_requested": args.model_sessions,
                "shared_model_session": args.shared_model_session,
                "seed_scheduler": "dynamic_atomic_queue",
                "model_session_topology": if args.shared_model_session {
                    "one_shared_bridge_with_worker_clients"
                } else {
                    "one_persistent_bridge_per_seed_worker"
                },
            }),
        );
        object.insert(
            "created_unix_seconds".to_owned(),
            json!(created_unix_seconds()?),
        );
        object.insert(
            "scientific_eligibility".to_owned(),
            json!(
                if args.model_manifest.is_some() && !args.allow_model_fallback {
                    "gumbel_selfplay_expert_iteration"
                } else {
                    "audit_only_unverified_or_uniform_model_fallback"
                }
            ),
        );
    }
    let metadata_json = canonical_json(&metadata);
    npz_writer::write_expert_tensor_npz(
        &args.out,
        npz_writer::ExpertTensorNpz {
            version: feature_tensors::EXPERT_SHARD_VERSION_V4,
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
            exact_afterstate_score_decomposition_active: Some(
                &shard.exact_afterstate_score_decomposition_active,
            ),
            active_seat: Some(&shard.active_seat),
            final_score_vector: &shard.final_score_vector,
            rank_vector: &shard.rank_vector,
            score_decomposition: &shard.score_decomposition,
            improved_policy: Some(&shard.improved_policy),
            search_root_value: Some(&shard.search_root_value),
            exact_endgame: Some(&shard.exact_endgame),
            record_count: shard.record_count,
            compression: args.tensor_compression,
        },
    )?;
    let checksum = sha256_file_hex(&args.out)?;
    write_expert_tensor_manifest(&args.manifest, args, &shard, &checksum, &metadata)?;
    Ok(shard.record_count)
}

/// Plays one seed to terminal with all four seats driven by Gumbel search,
/// returning the per-decision records plus the final `gumbel_game_done`
/// record. This is the shared per-seed core of `--gumbel-policy-game` and
/// `--gumbel-benchmark-batch`: any change here changes both modes
/// identically, which is what keeps their outputs equivalent.
fn play_gumbel_policy_game_seed(
    args: &Args,
    seed_u64: u64,
    chunk_bridge: &mut ChunkBridge,
    eval_cache: &mut EvalRowCache,
) -> Result<Vec<Value>> {
    let mut records = Vec::new();
    let config = GameConfig::research_aaaaa(args.player_count)?;
    let mut game = GameState::new(config, GameSeed::from_u64(seed_u64))
        .with_context(|| format!("creating gumbel policy game seed {seed_u64}"))?;
    let game_started = Instant::now();
    let mut ply_index = 0usize;
    while !game.is_game_over() {
        let decision_started = Instant::now();
        let cfg = gumbel_config_from_args(args, gumbel_search_seed(seed_u64, ply_index));
        let mut evaluator = BridgeLeafEvaluator {
            bridge: chunk_bridge,
            allow_model_fallback: args.allow_model_fallback,
            cache: eval_cache,
            tta_rotations: args.gumbel_tta,
        };
        let Some(decision) = gumbel::gumbel_search_for_state(
            &game,
            gumbel_root_menu_limit(args),
            &mut evaluator,
            &cfg,
        )?
        else {
            break;
        };
        let market_branches_searched = decision.market_branches_searched;
        let market_chance_samples = decision.market_chance_samples;
        let total_simulations_run = decision.total_simulations_run;
        let exact_endgame = decision.exact_endgame;
        let row = decision.row;
        let result = decision.result;
        let chosen = &row.afterstates[result.chosen_index];
        records.push(json!({
            "type": "gumbel_decision",
            "ruleset_id": RULESET_ID,
            "seed": seed_u64,
            "ply": ply_index,
            "active_seat": row.staged.current_player(),
            "action_count": row.afterstates.len(),
            "chosen_action_id": action_id(&chosen.candidate.action)?,
            "root_value": result.root_value,
            "simulations_run": result.simulations_run,
            "market_branches_searched": market_branches_searched,
            "market_chance_samples": market_chance_samples,
            "total_simulations_run": total_simulations_run,
            "exact_endgame": exact_endgame,
            "free_three_of_a_kind_choice": if row.prelude.replace_three_of_a_kind {
                "accept"
            } else if row.staged.market().three_of_a_kind().is_some() {
                "decline"
            } else {
                "not_available"
            },
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
        "ruleset_id": RULESET_ID,
        "seed": seed_u64,
        "scores": scores.iter().map(score_breakdown_json).collect::<Vec<_>>(),
        "decision_count": ply_index,
        "elapsed_seconds": game_started.elapsed().as_secs_f64(),
        "search": {
            "n_simulations": args.gumbel_n_simulations,
            "top_m": args.gumbel_top_m,
            "depth_rounds": args.gumbel_depth_rounds,
            "determinization_samples": args.gumbel_determinizations,
            "market_decision_samples": args.gumbel_market_decision_samples,
            "exact_endgame_turns": args.gumbel_exact_endgame_turns,
            "rollout_blend_weight": args.gumbel_blend_weight,
            "parallel_leaf_rollouts": args.gumbel_parallel_leaf_rollouts,
            "exploration": args.gumbel_exploration,
            "k_interior": args.k_interior,
        },
    }));
    eprintln!(
        "gumbel policy game seed {seed_u64} complete: {} decisions in {:.1}s",
        ply_index,
        game_started.elapsed().as_secs_f64()
    );
    Ok(records)
}

/// Persistent interactive suggestion server: reads JSONL requests carrying a
/// serialized `GameState` on stdin, runs the configured Gumbel search, and
/// answers with the ranked legal menu on stdout. Powers the web UI's
/// "champion" strength. Emits a `suggest_ready` line once the model bridge
/// is up so the parent can gate readiness.
fn run_gumbel_suggest_server(args: &Args) -> Result<()> {
    use std::io::{BufRead, Write};

    let mut chunk_bridge = ChunkBridge::Owned(model_state_worker_session(args)?);
    let mut eval_cache = EvalRowCache::new();
    let stdout = std::io::stdout();
    {
        let mut out = stdout.lock();
        serde_json::to_writer(&mut out, &json!({"type": "suggest_ready"}))?;
        out.write_all(b"\n")?;
        out.flush()?;
    }
    let stdin = std::io::stdin();
    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let request: Value = match serde_json::from_str(&line) {
            Ok(value) => value,
            Err(error) => {
                let mut out = stdout.lock();
                serde_json::to_writer(
                    &mut out,
                    &json!({"type": "suggest_error", "id": null, "error": format!("bad request json: {error}")}),
                )?;
                out.write_all(b"\n")?;
                out.flush()?;
                continue;
            }
        };
        let id = request.get("id").cloned().unwrap_or(Value::Null);
        let mut response =
            match suggest_for_request(args, &request, &mut chunk_bridge, &mut eval_cache) {
                Ok(value) => value,
                Err(error) => json!({"type": "suggest_error", "error": error.to_string()}),
            };
        response["id"] = id;
        let mut out = stdout.lock();
        serde_json::to_writer(&mut out, &response)?;
        out.write_all(b"\n")?;
        out.flush()?;
    }
    Ok(())
}

/// One suggestion: deserialize the game, search, return the full ranked menu
/// (actions in game-crate serde form, action-aligned Q/policy arrays).
fn suggest_for_request(
    args: &Args,
    request: &Value,
    bridge: &mut ChunkBridge,
    cache: &mut EvalRowCache,
) -> Result<Value> {
    let game: GameState = serde_json::from_value(
        request
            .get("game")
            .context("suggest request missing game")?
            .clone(),
    )
    .context("deserializing game state")?;
    // Deterministic per-position search seed: identical positions get
    // identical suggestions across requests and restarts.
    let public_hash = game.public_state().canonical_hash();
    let seed = u64::from_le_bytes(public_hash.as_bytes()[..8].try_into().expect("hash prefix"));
    let mut cfg = gumbel_config_from_args(args, seed);
    // Per-request search-shape overrides so one server (one loaded model)
    // can answer at several strength tiers.
    if let Some(n_simulations) = request.get("n_simulations").and_then(Value::as_u64) {
        cfg.n_simulations = (n_simulations as usize).clamp(1, 65_536);
    }
    if let Some(determinizations) = request.get("determinizations").and_then(Value::as_u64) {
        cfg.determinization_samples = (determinizations as usize).clamp(1, 256);
    }
    if let Some(top_m) = request.get("top_m").and_then(Value::as_u64) {
        cfg.top_m = (top_m as usize).clamp(1, 256);
    }
    let mut evaluator = BridgeLeafEvaluator {
        bridge,
        allow_model_fallback: args.allow_model_fallback,
        cache,
        tta_rotations: args.gumbel_tta,
    };
    let Some(decision) =
        gumbel::gumbel_search_for_state(&game, gumbel_root_menu_limit(args), &mut evaluator, &cfg)?
    else {
        return Ok(json!({"type": "suggest_response", "game_over": true}));
    };
    let market_branches_searched = decision.market_branches_searched;
    let market_chance_samples = decision.market_chance_samples;
    let total_simulations_run = decision.total_simulations_run;
    let exact_endgame = decision.exact_endgame;
    let row = decision.row;
    let result = decision.result;
    // Candidate actions are relative to their selected staged market. Merge
    // that explicit policy choice back into each returned compound turn so
    // applying it to the original game reproduces the searched branch.
    let actions = row
        .afterstates
        .iter()
        .map(|afterstate| {
            let mut action = afterstate.candidate.action.clone();
            action.replace_three_of_a_kind = row.prelude.replace_three_of_a_kind;
            action.wildlife_wipes = row.prelude.wildlife_wipes.clone();
            serde_json::to_value(&action)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(json!({
        "type": "suggest_response",
        "ruleset_id": RULESET_ID,
        "game_over": false,
        "chosen_index": result.chosen_index,
        "actions": actions,
        "completed_q": result.completed_q,
        "improved_policy": result.improved_policy,
        "visit_counts": result.visit_counts,
        "root_value": result.root_value,
        "market_branches_searched": market_branches_searched,
        "market_chance_samples": market_chance_samples,
        "total_simulations_run": total_simulations_run,
        "exact_endgame": exact_endgame,
        "free_three_of_a_kind_choice": if row.prelude.replace_three_of_a_kind {
            "accept"
        } else if row.staged.market().three_of_a_kind().is_some() {
            "decline"
        } else {
            "not_available"
        },
        "exact_afterstate_scores": row
            .afterstates
            .iter()
            .map(|afterstate| afterstate.exact_score_active)
            .collect::<Vec<_>>(),
    }))
}

fn run_gumbel_policy_game(args: &Args) -> Result<()> {
    let seed_end = args
        .first_seed
        .checked_add(args.seed_count)
        .context("seed range overflow")?;
    let mut chunk_bridge = ChunkBridge::Owned(model_state_worker_session(args)?);
    let mut eval_cache = EvalRowCache::new();
    let mut records = Vec::new();
    for seed_u64 in args.first_seed..seed_end {
        records.extend(play_gumbel_policy_game_seed(
            args,
            seed_u64,
            &mut chunk_bridge,
            &mut eval_cache,
        )?);
    }
    log_eval_dedup_summary(
        "gumbel policy game",
        eval_cache.stats.rows_requested,
        eval_cache.stats.rows_sent,
        eval_cache.stats.cache_hits,
    );
    write_jsonl(&args.out, &records)?;
    Ok(())
}

/// One `gumbel_decision` ledger row, as needed to replay the trajectory.
#[derive(Debug, Clone)]
struct LedgerDecisionRow {
    ply: u64,
    chosen_action_id: String,
    free_choice: String,
    active_seat: Option<u64>,
    action_count: Option<u64>,
}

/// Metadata for one reconstructed root, parallel to its `EvalRow`.
struct ReplayedRootMeta {
    ply: usize,
    chosen_index: usize,
    free_choice: String,
    /// The chosen action was only found on the uncapped legal menu
    /// (exact-endgame decisions search the full legal set).
    full_menu_fallback: bool,
}

/// Reads `gumbel_decision` rows from a decisions JSONL ledger, grouped by
/// seed and validated to be contiguous plies from 0.
fn read_ledger_decision_rows(path: &PathBuf) -> Result<BTreeMap<u64, Vec<LedgerDecisionRow>>> {
    let contents = std::fs::read_to_string(path)
        .with_context(|| format!("reading decisions ledger {}", path.display()))?;
    let mut by_seed: BTreeMap<u64, Vec<LedgerDecisionRow>> = BTreeMap::new();
    for (line_index, line) in contents.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let value: Value = serde_json::from_str(line)
            .with_context(|| format!("parsing ledger line {}", line_index + 1))?;
        if value.get("type").and_then(Value::as_str) != Some("gumbel_decision") {
            continue;
        }
        let seed = value
            .get("seed")
            .and_then(Value::as_u64)
            .with_context(|| format!("ledger line {} lacks a seed", line_index + 1))?;
        let row = LedgerDecisionRow {
            ply: value
                .get("ply")
                .and_then(Value::as_u64)
                .with_context(|| format!("ledger line {} lacks a ply", line_index + 1))?,
            chosen_action_id: value
                .get("chosen_action_id")
                .and_then(Value::as_str)
                .with_context(|| format!("ledger line {} lacks chosen_action_id", line_index + 1))?
                .to_owned(),
            free_choice: value
                .get("free_three_of_a_kind_choice")
                .and_then(Value::as_str)
                .unwrap_or("not_available")
                .to_owned(),
            active_seat: value.get("active_seat").and_then(Value::as_u64),
            action_count: value.get("action_count").and_then(Value::as_u64),
        };
        by_seed.entry(seed).or_default().push(row);
    }
    if by_seed.is_empty() {
        bail!("no gumbel_decision rows found in {}", path.display());
    }
    for (seed, rows) in &mut by_seed {
        rows.sort_by_key(|row| row.ply);
        for (index, row) in rows.iter().enumerate() {
            if row.ply != index as u64 {
                bail!("ledger seed {seed} has non-contiguous plies (missing ply {index})");
            }
        }
    }
    Ok(by_seed)
}

fn find_action_index(row: &gumbel::EvalRow, target: &str) -> Result<Option<usize>> {
    for (index, afterstate) in row.afterstates.iter().enumerate() {
        if action_id(&afterstate.candidate.action)? == target {
            return Ok(Some(index));
        }
    }
    Ok(None)
}

/// Rebuilds the root row the serving search saw (same greedy menu cap and
/// market prelude) and locates the ledger's chosen action on it. Falls back
/// to the uncapped legal menu (exact-endgame decisions search the full set).
fn reconstruct_row_with_chosen(
    game: &GameState,
    prelude: MarketPrelude,
    menu_limit: Option<usize>,
    chosen_action_id: &str,
) -> Result<(gumbel::EvalRow, usize, bool)> {
    let row = gumbel::eval_row_for_prelude(game, prelude.clone(), menu_limit)?
        .context("non-terminal ledger decision produced no eval row")?;
    if let Some(index) = find_action_index(&row, chosen_action_id)? {
        return Ok((row, index, false));
    }
    let full_row = gumbel::eval_row_for_prelude(game, prelude, None)?
        .context("non-terminal ledger decision produced no full-menu eval row")?;
    let index = find_action_index(&full_row, chosen_action_id)?
        .context("chosen action id absent even from the full legal menu")?;
    Ok((full_row, index, true))
}

/// Replays one seed's ledger decisions without any search: reconstructs each
/// root row, locates the chosen action, and advances to its afterstate.
/// Returns parallel (rows, metas) plus the final reached state. Fails closed
/// on any divergence from the ledger (seat or menu-size mismatch).
fn replay_ledger_seed(
    seed: u64,
    rows: &[LedgerDecisionRow],
    menu_limit: Option<usize>,
    player_count: u8,
) -> Result<(Vec<gumbel::EvalRow>, Vec<ReplayedRootMeta>, GameState)> {
    let config = GameConfig::research_aaaaa(player_count)?;
    let mut game = GameState::new(config, GameSeed::from_u64(seed))
        .with_context(|| format!("creating replay game for seed {seed}"))?;
    let mut eval_rows = Vec::with_capacity(rows.len());
    let mut metas = Vec::with_capacity(rows.len());
    for ledger_row in rows {
        if game.is_game_over() {
            bail!(
                "ledger seed {seed} ply {} follows a terminal state",
                ledger_row.ply
            );
        }
        let prelude = match ledger_row.free_choice.as_str() {
            "accept" => game
                .free_three_of_a_kind_choices()?
                .into_iter()
                .find(|choice| choice.replace_three_of_a_kind)
                .with_context(|| {
                    format!(
                        "ledger seed {seed} ply {} says accept but no accept choice is legal",
                        ledger_row.ply
                    )
                })?,
            "decline" | "not_available" => MarketPrelude::default(),
            other => bail!(
                "ledger seed {seed} ply {}: unknown free_three_of_a_kind_choice {other}",
                ledger_row.ply
            ),
        };
        let (row, chosen_index, full_menu_fallback) =
            reconstruct_row_with_chosen(&game, prelude, menu_limit, &ledger_row.chosen_action_id)
                .with_context(|| format!("reconstructing seed {seed} ply {}", ledger_row.ply))?;
        if let Some(expected_seat) = ledger_row.active_seat {
            let actual_seat = row.staged.current_player() as u64;
            if actual_seat != expected_seat {
                bail!(
                    "replay divergence at seed {seed} ply {}: seat {actual_seat} vs ledger {expected_seat}",
                    ledger_row.ply
                );
            }
        }
        if !full_menu_fallback {
            if let Some(expected_count) = ledger_row.action_count {
                let actual_count = row.afterstates.len() as u64;
                if actual_count != expected_count {
                    bail!(
                        "replay divergence at seed {seed} ply {}: {actual_count} actions vs ledger {expected_count}",
                        ledger_row.ply
                    );
                }
            }
        }
        let chosen = &row.afterstates[chosen_index];
        let next_game = chosen.state.clone();
        let truncated = chosen.apply_truncated;
        metas.push(ReplayedRootMeta {
            ply: ledger_row.ply as usize,
            chosen_index,
            free_choice: ledger_row.free_choice.clone(),
            full_menu_fallback,
        });
        eval_rows.push(row);
        game = next_game;
        if truncated {
            break;
        }
    }
    Ok((eval_rows, metas, game))
}

fn eval_rows_chunked(
    evaluator: &mut BridgeLeafEvaluator,
    rows: &[gumbel::EvalRow],
    chunk_size: usize,
) -> Result<Vec<gumbel::EvalOut>> {
    use gumbel::LeafEvaluator;
    let mut outs = Vec::with_capacity(rows.len());
    for slice in rows.chunks(chunk_size.max(1)) {
        outs.extend(evaluator.evaluate_batch(slice)?);
    }
    Ok(outs)
}

/// Argmax over model derived final Q, optionally excluding one index.
/// `None` when no eligible action remains (single-action menus).
fn best_q_index(derived_final_q: &[f64], excluding: Option<usize>) -> Option<usize> {
    let mut best: Option<usize> = None;
    for (index, q) in derived_final_q.iter().enumerate() {
        if Some(index) == excluding {
            continue;
        }
        if best.map_or(true, |current| *q > derived_final_q[current]) {
            best = Some(index);
        }
    }
    best
}

/// Estimated table total (sum over all seats' final scores) for one
/// afterstate: exact terminal score when the game (or a truncation) ends
/// there, otherwise the sum of the model value head's per-seat predictions.
fn afterstate_table_estimate(
    state: &GameState,
    menu_limit: Option<usize>,
    evaluator: &mut BridgeLeafEvaluator,
) -> Result<(f64, bool)> {
    use gumbel::LeafEvaluator;
    let row = if state.is_game_over() {
        None
    } else {
        gumbel::eval_row_for_prelude(state, MarketPrelude::default(), menu_limit)?
    };
    let Some(row) = row else {
        let exact: f64 = score_game(state)
            .iter()
            .map(|score| f64::from(score.total))
            .sum();
        return Ok((exact, true));
    };
    let eval = evaluator
        .evaluate_batch(std::slice::from_ref(&row))?
        .into_iter()
        .next()
        .context("afterstate evaluation returned no output")?;
    let values = eval.value_vector.context(
        "model value head (per-seat final predictions) is required for the \
         table-contention audit; refusing to run on a fallback bridge",
    )?;
    Ok((values.iter().sum(), false))
}

/// R1.1a contention audit: replays stored decision ledgers (no search) and,
/// for every decision, compares the chosen action against the best
/// alternative by model Q under the TABLE objective (sum of the value head's
/// per-seat predictions at each afterstate). Bounds the prize of cooperative
/// table optimization before any training. The alternative ranking uses
/// model derived Q (the search's completed-Q runner-up is not recoverable
/// from ledgers without re-searching) — read the output as a bound
/// estimator, not a gate.
fn run_table_contention_audit(args: &Args) -> Result<()> {
    let input = args
        .input
        .as_ref()
        .context("--table-contention-audit requires --in <decisions.jsonl>")?;
    let by_seed = read_ledger_decision_rows(input)?;
    let menu_limit = gumbel_root_menu_limit(args);
    let mut chunk_bridge = ChunkBridge::Owned(model_state_worker_session(args)?);
    let mut eval_cache = EvalRowCache::new();
    let started = Instant::now();
    let mut records = Vec::new();
    let mut decisions_audited = 0usize;
    let mut single_action_skipped = 0usize;
    let mut replay_skipped_seeds = Vec::new();
    let seed_count = by_seed.len();
    for (seed, ledger_rows) in &by_seed {
        // jobs12-generated ledgers can carry rare concurrency-divergent
        // seeds (2027071427 precedent); skip them loudly instead of
        // stranding the audit.
        let (eval_rows, metas, _final_state) =
            match replay_ledger_seed(*seed, ledger_rows, menu_limit, args.player_count) {
                Ok(replayed) => replayed,
                Err(error) => {
                    eprintln!("[table-contention-audit] SKIPPING seed {seed}: {error:#}");
                    replay_skipped_seeds.push(*seed);
                    continue;
                }
            };
        let mut evaluator = BridgeLeafEvaluator {
            bridge: &mut chunk_bridge,
            allow_model_fallback: args.allow_model_fallback,
            cache: &mut eval_cache,
            tta_rotations: args.gumbel_tta,
        };
        let root_evals = eval_rows_chunked(&mut evaluator, &eval_rows, 32)?;
        for ((row, meta), root_eval) in eval_rows.iter().zip(&metas).zip(&root_evals) {
            let derived_q = &root_eval.derived_final_q;
            if derived_q.len() != row.afterstates.len() {
                bail!("root eval misaligned with menu at seed {seed} ply {}", meta.ply);
            }
            let Some(runner_index) = best_q_index(derived_q, Some(meta.chosen_index)) else {
                single_action_skipped += 1;
                continue;
            };
            let model_best_index = best_q_index(derived_q, None)
                .context("non-empty menu must have a model-best action")?;
            let chosen_after = &row.afterstates[meta.chosen_index];
            let runner_after = &row.afterstates[runner_index];
            let (chosen_table, chosen_exact) =
                afterstate_table_estimate(&chosen_after.state, menu_limit, &mut evaluator)?;
            let (runner_table, runner_exact) =
                afterstate_table_estimate(&runner_after.state, menu_limit, &mut evaluator)?;
            records.push(json!({
                "type": "contention_decision",
                "ruleset_id": RULESET_ID,
                "seed": seed,
                "ply": meta.ply,
                "active_seat": row.staged.current_player(),
                "action_count": row.afterstates.len(),
                "free_three_of_a_kind_choice": meta.free_choice,
                "full_menu_fallback": meta.full_menu_fallback,
                "chosen": {
                    "index": meta.chosen_index,
                    "action_id": action_id(&chosen_after.candidate.action)?,
                    "model_q": derived_q[meta.chosen_index],
                },
                "model_best": {
                    "index": model_best_index,
                    "action_id": action_id(&row.afterstates[model_best_index].candidate.action)?,
                    "model_q": derived_q[model_best_index],
                },
                "runner": {
                    "index": runner_index,
                    "action_id": action_id(&runner_after.candidate.action)?,
                    "model_q": derived_q[runner_index],
                },
                "chosen_table": chosen_table,
                "chosen_table_exact": chosen_exact,
                "runner_table": runner_table,
                "runner_table_exact": runner_exact,
                "table_delta_runner_minus_chosen": runner_table - chosen_table,
                "own_q_sacrifice_chosen_minus_runner": derived_q[meta.chosen_index]
                    - derived_q[runner_index],
            }));
            decisions_audited += 1;
        }
        eprintln!(
            "[table-contention-audit] seed {seed} complete ({decisions_audited} decisions so far)"
        );
    }
    if decisions_audited == 0 {
        bail!("contention audit produced no audited decisions (every seed skipped?)");
    }
    records.push(json!({
        "type": "contention_summary",
        "ruleset_id": RULESET_ID,
        "seeds": seed_count,
        "decisions_audited": decisions_audited,
        "single_action_skipped": single_action_skipped,
        "replay_skipped_seeds": replay_skipped_seeds,
        "menu_limit": menu_limit,
        "elapsed_seconds": started.elapsed().as_secs_f64(),
    }));
    write_jsonl(&args.out, &records)?;
    eprintln!(
        "[table-contention-audit] wrote {decisions_audited} decision audits from {seed_count} seeds to {}",
        args.out.display()
    );
    Ok(())
}

fn stability_probe_seed(seed: u64, ply: usize, repeat: usize) -> u64 {
    gumbel::splitmix64(
        seed ^ gumbel::splitmix64(ply as u64 ^ 0x57ab_1e26)
            ^ gumbel::splitmix64(repeat as u64 ^ 0x0be5_5eed),
    )
}

fn stability_top_actions(
    row: &gumbel::EvalRow,
    result: &gumbel::GumbelSearchResult,
    limit: usize,
    visited_only: bool,
) -> Result<Vec<Value>> {
    let mut indexes: Vec<usize> = (0..result.completed_q.len())
        .filter(|&index| !visited_only || result.visit_counts[index] > 0)
        .collect();
    indexes.sort_by(|&left, &right| {
        result.completed_q[right]
            .partial_cmp(&result.completed_q[left])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    indexes
        .into_iter()
        .take(limit)
        .map(|index| {
            Ok(json!({
                "index": index,
                "action_id": action_id(&row.afterstates[index].candidate.action)?,
                "completed_q": result.completed_q[index],
                "visits": result.visit_counts[index],
            }))
        })
        .collect()
}

/// R0.2 offline check: replays stored ledgers to sample real serving roots,
/// then re-runs the root search repeatedly with fresh search seeds, once
/// per repeat with the incumbent unpaired rollout streams and once with
/// paired (CRN) rollout streams. Equal repeat indexes share one search seed
/// across the two variants, so worlds match and only the rollout-noise
/// structure differs. Exact-K1 frontier roots are skipped (served exactly).
fn run_search_stability_probe(args: &Args) -> Result<()> {
    let input = args
        .input
        .as_ref()
        .context("--search-stability-probe requires --in <decisions.jsonl>")?;
    if args.gumbel_blend_weight < 1.0 && args.rollout_top_k <= 1 {
        bail!(
            "--search-stability-probe with blended rollouts requires --rollout-top-k > 1: \
             top-k-1 greedy rollouts are deterministic, so the rollout RNG stream is never \
             consulted and the paired/unpaired comparison is vacuous (serving uses \
             --max-actions 64 --rollout-top-k 4)"
        );
    }
    let by_seed = read_ledger_decision_rows(input)?;
    let menu_limit = gumbel_root_menu_limit(args);
    let stride = args.probe_stride.max(1);
    let repeats = args.probe_repeats.max(2);
    let max_roots = args.probe_max_roots.max(1);
    let mut chunk_bridge = ChunkBridge::Owned(model_state_worker_session(args)?);
    let mut eval_cache = EvalRowCache::new();
    let started = Instant::now();
    let mut records = Vec::new();
    let mut sampled_roots = 0usize;
    let mut decision_index = 0usize;
    let mut replay_skipped_seeds = Vec::new();
    'seeds: for (seed, ledger_rows) in &by_seed {
        // Same skip-and-count policy as the contention audit for rare
        // concurrency-divergent ledger seeds.
        let (eval_rows, metas, _final_state) =
            match replay_ledger_seed(*seed, ledger_rows, menu_limit, args.player_count) {
                Ok(replayed) => replayed,
                Err(error) => {
                    eprintln!("[search-stability-probe] SKIPPING seed {seed}: {error:#}");
                    replay_skipped_seeds.push(*seed);
                    continue;
                }
            };
        for (row, meta) in eval_rows.iter().zip(&metas) {
            let is_exact_frontier = row
                .staged
                .turns_remaining_for_player(row.staged.current_player())
                == 1;
            let selected = !is_exact_frontier
                && row.afterstates.len() > 1
                && decision_index % stride == 0;
            decision_index += 1;
            if !selected {
                continue;
            }
            for paired_rollouts in [false, true] {
                for repeat in 0..repeats {
                    let mut cfg =
                        gumbel_config_from_args(args, stability_probe_seed(*seed, meta.ply, repeat));
                    cfg.paired_rollouts = paired_rollouts;
                    let mut evaluator = BridgeLeafEvaluator {
                        bridge: &mut chunk_bridge,
                        allow_model_fallback: args.allow_model_fallback,
                        cache: &mut eval_cache,
                        tta_rotations: args.gumbel_tta,
                    };
                    let result = gumbel::gumbel_search(row, &mut evaluator, &cfg)
                        .with_context(|| {
                            format!(
                                "stability search at seed {seed} ply {} repeat {repeat}",
                                meta.ply
                            )
                        })?;
                    records.push(json!({
                        "type": "stability_search",
                        "ruleset_id": RULESET_ID,
                        "seed": seed,
                        "ply": meta.ply,
                        "action_count": row.afterstates.len(),
                        "paired_rollouts": paired_rollouts,
                        "repeat": repeat,
                        "search_seed": cfg.search_seed,
                        "chosen_index": result.chosen_index,
                        "chosen_action_id":
                            action_id(&row.afterstates[result.chosen_index].candidate.action)?,
                        "simulations_run": result.simulations_run,
                        "top_overall": stability_top_actions(row, &result, 3, false)?,
                        "top_visited": stability_top_actions(row, &result, 3, true)?,
                    }));
                }
            }
            sampled_roots += 1;
            if sampled_roots % 10 == 0 {
                eprintln!(
                    "[search-stability-probe] {sampled_roots}/{max_roots} roots ({:.1}s elapsed)",
                    started.elapsed().as_secs_f64()
                );
            }
            if sampled_roots >= max_roots {
                break 'seeds;
            }
        }
    }
    if sampled_roots == 0 {
        bail!("stability probe sampled no roots (ledger too small or all exact-frontier)");
    }
    records.push(json!({
        "type": "stability_summary",
        "ruleset_id": RULESET_ID,
        "sampled_roots": sampled_roots,
        "repeats_per_variant": repeats,
        "stride": stride,
        "replay_skipped_seeds": replay_skipped_seeds,
        "menu_limit": menu_limit,
        "search": gumbel_search_probe_settings(args),
        "elapsed_seconds": started.elapsed().as_secs_f64(),
    }));
    write_jsonl(&args.out, &records)?;
    eprintln!(
        "[search-stability-probe] wrote {sampled_roots} roots x 2 variants x {repeats} repeats to {}",
        args.out.display()
    );
    Ok(())
}

fn gumbel_search_probe_settings(args: &Args) -> Value {
    json!({
        "n_simulations": args.gumbel_n_simulations,
        "top_m": args.gumbel_top_m,
        "depth_rounds": args.gumbel_depth_rounds,
        "determinizations": args.gumbel_determinizations,
        "rollout_blend_weight": args.gumbel_blend_weight,
        "rollout_max_actions": args.max_actions,
        "rollout_top_k": args.rollout_top_k,
        "k_interior": args.k_interior,
        "c_visit": args.gumbel_c_visit,
        "c_scale": args.gumbel_c_scale,
    })
}

/// R2.1 puzzle bank / screen mode: replays a decisions ledger and resolves
/// every stride-selected root with the configured search, worker-pooled
/// across ledger seeds against one shared bridge (the saturation pattern —
/// AGENTS.md 2026-07-11). One JSONL shard per seed lands under
/// `--output-dir`. With mega-budget flags this generates the frozen bank
/// (use `--probe-repeats 2+` to average away value noise); with candidate
/// serving flags and `--probe-repeats 1` the same mode produces a screen
/// run, scored against the bank by `cascadiav3.analyze_puzzle_screen`.
/// Exact-K1 frontier roots and single-action menus are excluded, matching
/// the stability probe. Divergent ledger seeds are skipped loudly.
fn run_puzzle_bank(args: &Args) -> Result<()> {
    let input = args
        .input
        .as_ref()
        .context("--puzzle-bank requires --in <decisions.jsonl>")?;
    let output_dir = args
        .output_dir
        .as_ref()
        .context("--puzzle-bank requires --output-dir")?;
    std::fs::create_dir_all(output_dir)
        .with_context(|| format!("creating --output-dir {}", output_dir.display()))?;
    let by_seed = read_ledger_decision_rows(input)?;
    // Trajectory reconstruction must use the cap the ledger was recorded
    // with. A capped --gumbel-root-menu applies to both replay and
    // resolution (the normal case); the FULL menu (0) applies only to the
    // RESOLVED roots (coverage audits), with replay falling back to the
    // serving-era cap of 256.
    const LEDGER_ROOT_MENU_CAP: usize = 256;
    let search_menu_limit = gumbel_root_menu_limit(args);
    let replay_menu_limit = search_menu_limit.or(Some(LEDGER_ROOT_MENU_CAP));
    let rebuild_search_menu = search_menu_limit != replay_menu_limit;
    let stride = args.probe_stride.max(1);
    let repeats = args.probe_repeats.max(1);
    let started = Instant::now();
    let seeds: Vec<u64> = by_seed.keys().copied().collect();
    let total_seeds = seeds.len() as u64;
    let completed_seeds = AtomicU64::new(0);
    let resolved_roots = AtomicU64::new(0);
    let skipped_seeds = Mutex::new(Vec::<u64>::new());
    let target_workers = args
        .model_sessions
        .unwrap_or_else(|| rayon::current_num_threads().max(1))
        .max(1);
    let shared_bridge = if args.shared_model_session {
        let command = args
            .model_service
            .as_ref()
            .context("--shared-model-session requires --model-service")?;
        Some(SharedBridge::spawn(
            command,
            &BridgeConfig::from_args(args),
            model_bridge::shared_row_cap(),
        )?)
    } else {
        None
    };
    let (_, workers) = run_dynamic_seed_workers(
        &seeds,
        target_workers,
        |_| {
            let bridge = match &shared_bridge {
                Some(shared) => ChunkBridge::Shared(shared.client()),
                None => ChunkBridge::Owned(model_state_worker_session(args)?),
            };
            Ok(GumbelSeedWorker {
                bridge,
                eval_cache: EvalRowCache::new(),
            })
        },
        |worker, seed| {
            let ledger_rows = &by_seed[&seed];
            let (eval_rows, metas, _final_state) =
                match replay_ledger_seed(seed, ledger_rows, replay_menu_limit, args.player_count)
                {
                    Ok(replayed) => replayed,
                    Err(error) => {
                        eprintln!("[puzzle-bank] SKIPPING seed {seed}: {error:#}");
                        skipped_seeds.lock().expect("skip list lock").push(seed);
                        return Ok(());
                    }
                };
            let mut records = Vec::new();
            for (index, (replayed_row, meta)) in eval_rows.iter().zip(&metas).enumerate() {
                let is_exact_frontier = replayed_row
                    .staged
                    .turns_remaining_for_player(replayed_row.staged.current_player())
                    == 1;
                if is_exact_frontier || replayed_row.afterstates.len() < 2 || index % stride != 0
                {
                    continue;
                }
                let rebuilt_row;
                let row: &gumbel::EvalRow = if rebuild_search_menu {
                    rebuilt_row = gumbel::rebuild_row_with_menu(replayed_row, search_menu_limit)
                        .with_context(|| {
                            format!("rebuilding menu at seed {seed} ply {}", meta.ply)
                        })?;
                    &rebuilt_row
                } else {
                    replayed_row
                };
                let action_count = row.afterstates.len();
                let mut mean_completed_q = vec![0.0_f64; action_count];
                let mut total_visits = vec![0_u64; action_count];
                let mut repeat_chosen = Vec::with_capacity(repeats);
                for repeat in 0..repeats {
                    let cfg =
                        gumbel_config_from_args(args, stability_probe_seed(seed, meta.ply, repeat));
                    let mut evaluator = BridgeLeafEvaluator {
                        bridge: &mut worker.bridge,
                        allow_model_fallback: args.allow_model_fallback,
                        cache: &mut worker.eval_cache,
                        tta_rotations: args.gumbel_tta,
                    };
                    let result = gumbel::gumbel_search(row, &mut evaluator, &cfg)
                        .with_context(|| {
                            format!("puzzle search at seed {seed} ply {} repeat {repeat}", meta.ply)
                        })?;
                    for action in 0..action_count {
                        mean_completed_q[action] += result.completed_q[action];
                        total_visits[action] += u64::from(result.visit_counts[action]);
                    }
                    repeat_chosen.push(result.chosen_index);
                }
                for value in &mut mean_completed_q {
                    *value /= repeats as f64;
                }
                let action_ids = row
                    .afterstates
                    .iter()
                    .map(|afterstate| action_id(&afterstate.candidate.action))
                    .collect::<Result<Vec<_>>>()?;
                let first_choice = repeat_chosen[0];
                let repeat_agreement = repeat_chosen
                    .iter()
                    .filter(|&&chosen| chosen == first_choice)
                    .count() as f64
                    / repeats as f64;
                records.push(json!({
                    "type": "puzzle_root",
                    "ruleset_id": RULESET_ID,
                    "seed": seed,
                    "ply": meta.ply,
                    "active_seat": row.staged.current_player(),
                    "action_count": action_count,
                    "ledger_chosen_action_id": ledger_rows[meta.ply].chosen_action_id,
                    "action_ids": action_ids,
                    "mean_completed_q": mean_completed_q,
                    "total_visits": total_visits,
                    "repeat_chosen_indexes": repeat_chosen,
                    "repeat_agreement": repeat_agreement,
                    "repeats": repeats,
                    "search": gumbel_search_probe_settings(args),
                }));
                resolved_roots.fetch_add(1, Ordering::Relaxed);
            }
            let shard_path = output_dir.join(format!("puzzle_seed_{seed}.jsonl"));
            write_jsonl(&shard_path, &records)
                .with_context(|| format!("writing puzzle shard for seed {seed}"))?;
            let done = completed_seeds.fetch_add(1, Ordering::Relaxed) + 1;
            eprintln!(
                "[puzzle-bank] seed {seed} complete ({done}/{total_seeds} seeds, {} roots, {:.1}s elapsed)",
                resolved_roots.load(Ordering::Relaxed),
                started.elapsed().as_secs_f64()
            );
            Ok(())
        },
    )?;
    let mut eval_rows_requested = AtomicU64::new(0);
    let mut eval_rows_sent = AtomicU64::new(0);
    let mut eval_cache_hits = AtomicU64::new(0);
    for worker in &workers {
        worker.eval_cache.stats.accumulate_into(
            &eval_rows_requested,
            &eval_rows_sent,
            &eval_cache_hits,
        );
    }
    log_eval_dedup_summary(
        "puzzle bank",
        *eval_rows_requested.get_mut(),
        *eval_rows_sent.get_mut(),
        *eval_cache_hits.get_mut(),
    );
    let skipped = skipped_seeds.into_inner().expect("skip list lock");
    let total_roots = resolved_roots.load(Ordering::Relaxed);
    if total_roots == 0 {
        bail!("puzzle bank resolved no roots (ledger too small or all excluded)");
    }
    let manifest_path = output_dir.join("puzzle_bank_manifest.json");
    let manifest = json!({
        "type": "puzzle_bank_manifest",
        "ruleset_id": RULESET_ID,
        "source_revision": args.source_revision,
        "ledger": input.display().to_string(),
        "seeds": seeds.len(),
        "replay_skipped_seeds": skipped,
        "resolved_roots": total_roots,
        "stride": stride,
        "repeats": repeats,
        "replay_menu_limit": replay_menu_limit,
        "search_menu_limit": search_menu_limit,
        "search": gumbel_search_probe_settings(args),
        "elapsed_seconds": started.elapsed().as_secs_f64(),
    });
    std::fs::write(&manifest_path, format!("{}\n", canonical_json(&manifest)))
        .with_context(|| format!("writing {}", manifest_path.display()))?;
    eprintln!(
        "[puzzle-bank] complete: {total_roots} roots across {} seeds -> {}",
        seeds.len(),
        output_dir.display()
    );
    Ok(())
}

/// Many complete Gumbel policy games in one process sharing model bridge
/// capacity: a dynamic seed queue, `--model-sessions` persistent parallel
/// game workers, and with
/// `--shared-model-session` one aggregated bridge (one CUDA context) serving
/// every worker. Per-seed search/record behavior is identical to
/// `--gumbel-policy-game` (exploration stays off unless explicitly enabled);
/// each seed's decisions + done records land in
/// `<output-dir>/gumbel_game_seed_<seed>.jsonl`. Any seed failure aborts the
/// whole run with the failing seed named in the error.
fn run_gumbel_benchmark_batch(args: &Args) -> Result<()> {
    let output_dir = args
        .output_dir
        .as_ref()
        .context("--gumbel-benchmark-batch requires --output-dir")?;
    std::fs::create_dir_all(output_dir)
        .with_context(|| format!("creating --output-dir {}", output_dir.display()))?;
    let seed_end = args
        .first_seed
        .checked_add(args.seed_count)
        .context("seed range overflow")?;
    let started = Instant::now();
    let total_seeds = args.seed_count;
    let completed_seeds = AtomicU64::new(0);
    let completed_records = AtomicU64::new(0);
    let eval_rows_requested = AtomicU64::new(0);
    let eval_rows_sent = AtomicU64::new(0);
    let eval_cache_hits = AtomicU64::new(0);
    let seeds = (args.first_seed..seed_end).collect::<Vec<_>>();
    let target_workers = args
        .model_sessions
        .unwrap_or_else(|| rayon::current_num_threads().max(1))
        .max(1);
    let shared_bridge = if args.shared_model_session {
        let command = args
            .model_service
            .as_ref()
            .context("--shared-model-session requires --model-service")?;
        Some(SharedBridge::spawn(
            command,
            &BridgeConfig::from_args(args),
            model_bridge::shared_row_cap(),
        )?)
    } else {
        None
    };
    let (_, workers) = run_dynamic_seed_workers(
        &seeds,
        target_workers,
        |_| {
            let bridge = match &shared_bridge {
                Some(shared) => ChunkBridge::Shared(shared.client()),
                None => ChunkBridge::Owned(model_state_worker_session(args)?),
            };
            Ok(GumbelSeedWorker {
                bridge,
                eval_cache: EvalRowCache::new(),
            })
        },
        |worker, seed_u64| {
            let records = play_gumbel_policy_game_seed(
                args,
                seed_u64,
                &mut worker.bridge,
                &mut worker.eval_cache,
            )
            .with_context(|| format!("gumbel benchmark batch seed {seed_u64} failed"))?;
            let seed_path = output_dir.join(format!("gumbel_game_seed_{seed_u64}.jsonl"));
            write_jsonl(&seed_path, &records).with_context(|| {
                format!("writing gumbel benchmark batch seed {seed_u64} output")
            })?;
            let done = completed_seeds.fetch_add(1, Ordering::Relaxed) + 1;
            let records_done = completed_records.fetch_add(records.len() as u64, Ordering::Relaxed)
                + records.len() as u64;
            log_seed_export_progress(
                "gumbel benchmark batch",
                done,
                total_seeds,
                records_done,
                started,
            );
            Ok(())
        },
    )?;
    for worker in &workers {
        worker.eval_cache.stats.accumulate_into(
            &eval_rows_requested,
            &eval_rows_sent,
            &eval_cache_hits,
        );
    }
    log_eval_dedup_summary(
        "gumbel benchmark batch",
        eval_rows_requested.load(Ordering::Relaxed),
        eval_rows_sent.load(Ordering::Relaxed),
        eval_cache_hits.load(Ordering::Relaxed),
    );
    let elapsed = started.elapsed().as_secs_f64().max(1.0e-9);
    eprintln!(
        "[real-root-exporter] gumbel benchmark batch: {total_seeds} seeds complete in {elapsed:.1}s ({:.2} games/h), dynamic_seed_queue workers={}, into {}",
        total_seeds as f64 * 3600.0 / elapsed,
        workers.len(),
        output_dir.display(),
    );
    Ok(())
}

fn build_greedy_policy_root_record(
    game: &GameState,
    seed_u64: u64,
    ply_index: usize,
    args: &Args,
) -> Result<(Value, String)> {
    let active_seat = game.current_player();
    let (prelude, staged) = greedy_market_choice(game, Some(args.max_actions))?;
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
    let (prelude, staged) = greedy_market_choice(game, Some(args.max_actions))?;
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
    exact_score_decomposition_active: [f64; 3],
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
                exact_score_decomposition_active: score_components(&scores[active_seat]),
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
                let (terminal, continuation_truncated) =
                    complete_with_sampled_greedy(next, max_actions, rollout_top_k, &mut rng, None)?;
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
            "ruleset_id": RULESET_ID,
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
            "ruleset_id": RULESET_ID,
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
            "free_three_of_a_kind_choice": root_bundle.free_three_of_a_kind_choice,
        });
        writeln!(stdout, "{}", canonical_json(&decision_record))?;
        stdout.flush()?;
        decision_records.push(decision_record);
    }

    let scores = score_game(&game);
    let done_message = json!({
        "type": "done",
        "ruleset_id": RULESET_ID,
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
    free_three_of_a_kind_choice: &'static str,
}

fn build_interactive_root(
    game: &GameState,
    seed_u64: u64,
    ply_index: usize,
    args: &Args,
) -> Result<InteractiveRoot> {
    let active_seat = game.current_player();
    let (prelude, staged) = greedy_market_choice(game, Some(args.max_actions))?;
    let free_three_of_a_kind_available = game.free_three_of_a_kind_choices()?.len() > 1;
    let free_three_of_a_kind_choice = if prelude.replace_three_of_a_kind {
        "accept"
    } else if free_three_of_a_kind_available {
        "decline"
    } else {
        "not_available"
    };
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
    let score_means = current_scores.iter().map(score_mean).collect::<Vec<_>>();
    let uniform_prior = 1.0 / action_count as f64;
    let public_hash = staged.public_state().canonical_hash();
    let mut record = json!({
        "schema_id": SCHEMA_ID,
        "ruleset_id": RULESET_ID,
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
        free_three_of_a_kind_choice,
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

/// Chooses the optional free three-of-a-kind branch with the same immediate
/// greedy policy used by legacy corpus/rollout modes, then returns the staged
/// state those modes require for feature construction. Gumbel modes use the
/// stronger search-valued branch decision in `gumbel_search_for_state`.
fn greedy_market_choice(
    game: &GameState,
    limit: Option<usize>,
) -> Result<(MarketPrelude, GameState)> {
    let ranked = rank_greedy_actions_with_market_choice(game, limit)?;
    let action = ranked.first().context("no legal market-choice action")?;
    let prelude = action.action.prelude();
    let staged = game.preview_market_prelude(&prelude)?;
    Ok((prelude, staged))
}

fn advance_selected_action(
    game: &GameState,
    max_actions: usize,
    selected_action_id: &str,
) -> Result<GameState> {
    let (_prelude, staged) = greedy_market_choice(game, Some(max_actions))?;
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
    // Ranking buffers reused across every ply of this rollout.
    let mut scratch = GreedyRankScratch::default();
    while !game.is_game_over() {
        // A three-of-a-kind produces two complete-turn branches. Rank them
        // together so the rollout policy may accept or decline; retain the
        // scratch-buffer hot path when no market choice exists.
        let candidates = if game.market().three_of_a_kind().is_some() {
            rank_greedy_actions_with_market_choice(&game, Some(max_actions))
        } else {
            rank_greedy_actions_with_scratch(
                &game,
                &MarketPrelude::default(),
                Some(max_actions),
                &mut scratch,
            )
        };
        let candidates = match candidates {
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
            let [wildlife, habitat, nature_tokens] = score_components(score);
            means[seat].wildlife += wildlife;
            means[seat].habitat += habitat;
            means[seat].nature_tokens += nature_tokens;
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

fn expert_tensor_mode_contract(args: &Args) -> (&'static str, &'static str, &'static str) {
    match args.mode {
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
            "Rust-native packed v4 expert tensor shard from all-seat Gumbel self-play over determinized hidden states; per-action targets are search completed-Q values, exact category afterstates ground structured score-to-go, improved_policy is the Gumbel policy-improvement target, exact_endgame is explicit per root, and value labels are real terminal outcomes.",
        ),
        _ => (
            "expert_tensor_corpus",
            "expert_iteration_bootstrap_tensor_pretraining",
            "Rust-native packed expert tensor shard; JSONL audit can be generated separately, but trainer scale path reads this NPZ directly.",
        ),
    }
}

fn created_unix_seconds() -> Result<u64> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system time predates Unix epoch")?
        .as_secs())
}

fn file_identity(path: &Path) -> Result<Value> {
    let metadata = std::fs::metadata(path)
        .with_context(|| format!("reading artifact metadata {}", path.display()))?;
    Ok(json!({
        "path": path.display().to_string(),
        "bytes": metadata.len(),
        "sha256": sha256_file_hex(path)?,
    }))
}

fn resolve_manifest_weights_path(manifest_path: &Path, raw_weights: &str) -> Result<PathBuf> {
    let raw_path = PathBuf::from(raw_weights);
    if raw_path.is_absolute() {
        return Ok(raw_path);
    }
    let cwd = std::env::current_dir().context("reading current directory")?;
    let mut candidates = vec![cwd.join(&raw_path)];
    for ancestor in manifest_path.ancestors() {
        if ancestor.file_name().and_then(|name| name.to_str()) == Some("cascadiav3") {
            if let Some(project_root) = ancestor.parent() {
                let candidate = project_root.join(&raw_path);
                if !candidates.contains(&candidate) {
                    candidates.push(candidate);
                }
            }
        }
    }
    if let Some(parent) = manifest_path.parent() {
        let candidate = parent.join(&raw_path);
        if !candidates.contains(&candidate) {
            candidates.push(candidate);
        }
    }
    candidates
        .iter()
        .find(|candidate| candidate.exists())
        .cloned()
        .with_context(|| {
            format!(
                "checkpoint weights {raw_weights:?} not found; tried {}",
                candidates
                    .iter()
                    .map(|candidate| candidate.display().to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })
}

fn model_artifact_identity(args: &Args) -> Result<Value> {
    let Some(raw_manifest_path) = args.model_manifest.as_ref() else {
        return Ok(json!({
            "kind": "unverified_or_uniform_fallback",
            "allow_model_fallback": args.allow_model_fallback,
            "service": args.model_service,
            "manifest": null,
            "weights": null,
        }));
    };
    let manifest_path = if raw_manifest_path.is_absolute() {
        raw_manifest_path.clone()
    } else {
        std::env::current_dir()
            .context("reading current directory")?
            .join(raw_manifest_path)
    };
    let manifest_bytes = std::fs::read(&manifest_path)
        .with_context(|| format!("reading model manifest {}", manifest_path.display()))?;
    let manifest: Value = serde_json::from_slice(&manifest_bytes)
        .with_context(|| format!("parsing model manifest {}", manifest_path.display()))?;
    let raw_weights = manifest
        .get("weights")
        .and_then(Value::as_str)
        .context("model manifest is missing string weights")?;
    let weights_path = resolve_manifest_weights_path(&manifest_path, raw_weights)?;
    Ok(json!({
        "kind": "checkpoint_manifest_and_weights",
        "allow_model_fallback": args.allow_model_fallback,
        "service": args.model_service,
        "manifest": file_identity(&manifest_path)?,
        "weights": file_identity(&weights_path)?,
        "checkpoint_tag": manifest.get("checkpoint_tag"),
        "step": manifest.get("step"),
        "model_name": manifest.pointer("/config/model_name"),
        "model_size": manifest.pointer("/config/model_size"),
        "q_quantiles": manifest.pointer("/config/q_quantiles"),
        "q_decomposition": manifest.pointer("/config/q_decomposition"),
    }))
}

fn generator_artifact_identity() -> Result<Value> {
    let executable = std::env::current_exe().context("resolving exporter executable")?;
    let mut identity = file_identity(&executable)?;
    identity["kind"] = json!("cascadiav3-real-root-exporter");
    Ok(identity)
}

fn expert_tensor_shard_metadata(args: &Args, shard: &ExpertTensorShardData) -> Value {
    let (mode_name, _, _) = expert_tensor_mode_contract(args);
    let mut targets = vec![
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
        "score_decomposition",
    ];
    if args.mode == Mode::GumbelSelfplayTensorCorpus {
        targets.extend([
            "improved_policy",
            "search_root_value",
            "exact_endgame",
            "active_seat",
            "exact_afterstate_score_decomposition_active",
        ]);
    }
    json!({
        "version": EXPERT_SHARD_VERSION,
        "schema_id": EXPERT_TENSOR_SCHEMA_ID,
        "ruleset_id": RULESET_ID,
        "mode": mode_name,
        "source_revision": args.source_revision,
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
            "first_seed={},seed_count={},plies_per_seed={},max_actions={},rollouts_per_action={},rollout_top_k={},mode={}",
            args.first_seed,
            args.seed_count,
            args.plies_per_seed,
            args.max_actions,
            args.rollouts_per_action,
            args.rollout_top_k,
            mode_name,
        ),
        "canonical_model_inputs": [
            "public_token_features",
            "semantic_public_token_action_features",
            "sparse_relation_edges"
        ],
        "canonical_targets": targets,
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
    metadata: &Value,
) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let (mode_name, default_scientific_eligibility, notes) = expert_tensor_mode_contract(args);
    let scientific_eligibility = if args.mode == Mode::GumbelSelfplayTensorCorpus
        && (args.model_manifest.is_none() || args.allow_model_fallback)
    {
        "audit_only_unverified_or_uniform_model_fallback"
    } else {
        default_scientific_eligibility
    };
    let (schema_id, version) = if args.mode == Mode::GumbelSelfplayTensorCorpus {
        (EXPERT_TENSOR_SCHEMA_ID_V4, EXPERT_SHARD_VERSION_V4)
    } else {
        (EXPERT_TENSOR_SCHEMA_ID, EXPERT_SHARD_VERSION)
    };
    let manifest_created_unix_seconds = metadata
        .get("created_unix_seconds")
        .and_then(Value::as_u64)
        .map_or_else(created_unix_seconds, Ok)?;
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
        "created_unix_seconds": manifest_created_unix_seconds,
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
        "metadata": metadata,
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

fn sha256_file_hex(path: &Path) -> Result<String> {
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
            gumbel_peek: false,
            gumbel_table_total: false,
            gumbel_table_native_q: false,
            gumbel_leaf_softmix: None,
            gumbel_c_visit: 50.0,
            gumbel_c_scale: 1.0,
            gumbel_sigma_norm: gumbel::SigmaNormalization::MinMax,
            gumbel_paired_rollouts: false,
            gumbel_ghost_opponents: false,
            gumbel_q_bias_correction: false,
            gumbel_lcb_c: 0.0,
            gumbel_refresh_sample_divisor: 1,
            gumbel_tta: 1,
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
            source_revision: None,
            model_timeout_ms: 1_000,
            rollout_determinize: false,
            gumbel_n_simulations: 8,
            gumbel_top_m: 4,
            gumbel_depth_rounds: 1,
            gumbel_determinizations: 2,
            gumbel_market_decision_samples: 2,
            gumbel_exact_endgame_turns: 0,
            gumbel_blend_weight: 1.0,
            gumbel_parallel_leaf_rollouts: false,
            gumbel_exploration: false,
            gumbel_max_root_actions: None,
            gumbel_root_menu: 64,
            k_interior: 6,
            model_sessions: None,
            shared_model_session: false,
            output_dir: None,
            probe_stride: 7,
            probe_repeats: 6,
            probe_max_roots: 100,
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
        let (_prelude, staged) = greedy_market_choice(&game, Some(4)).expect("staged");
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
            evaluate_candidate_rollouts(
                root,
                &afterstates,
                root.current_player(),
                7,
                3,
                4,
                3,
                2,
                true,
            )
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
        let (unmoved, _) = complete_with_sampled_greedy(game.clone(), 4, 2, &mut rng_zero, Some(0))
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
            // Rules 2026-07-09: optional three-of-a-kind refreshes are
            // public-information policy decisions before the chance draw.
            "24ca921ec767b442acbc5495c9fbacd8790beb0346c94b795625aaf8194e2b7a"
        );
        let legacy_shard = ExpertTensorShardData::from_records(&records)
            .expect("legacy expert records with active_seat remain v1-packable");
        assert_eq!(legacy_shard.structured_value_field_records, 0);
        assert!(legacy_shard.active_seat.is_empty());
        assert!(
            legacy_shard
                .exact_afterstate_score_decomposition_active
                .is_empty()
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
        let weights_path = tempdir.join("mock.weights");
        std::fs::write(&weights_path, b"deterministic mock weights").expect("mock weights");
        let model_manifest_path = tempdir.join("mock.manifest.json");
        std::fs::write(
            &model_manifest_path,
            canonical_json(&json!({
                "checkpoint_tag": "mock-checkpoint",
                "step": 17,
                "config": {
                    "model_name": "Mock-CascadiaFormer",
                    "model_size": "test",
                    "q_quantiles": 1,
                },
                "weights": "mock.weights",
            })),
        )
        .expect("mock manifest");
        args.mode = Mode::GumbelSelfplayTensorCorpus;
        args.out = tempdir.join("gumbel_tiny.npz");
        args.manifest = tempdir.join("gumbel_tiny_manifest.json");
        args.model_service = Some(mock_bridge_command());
        args.model_manifest = Some(model_manifest_path);
        args.allow_model_fallback = false;
        args.source_revision = Some("test-source-revision".to_owned());
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
    fn gumbel_selfplay_records_roundtrip_into_v4_shard() {
        let tempdir =
            std::env::temp_dir().join(format!("cascadia-gumbel-test-{}", std::process::id()));
        std::fs::create_dir_all(&tempdir).expect("tempdir");
        let args = gumbel_test_args(&tempdir);

        let session = model_state_worker_session(&args).expect("mock session");
        assert!(session.is_some(), "mock bridge session must spawn");
        let mut bridge = ChunkBridge::Owned(session);
        let mut eval_cache = EvalRowCache::new();
        let records = play_gumbel_selfplay_seed(&args, 2_026_070_600, &mut bridge, &mut eval_cache)
            .expect("selfplay seed plays");
        assert!(!records.is_empty());
        assert!(eval_cache.stats.rows_requested > 0);
        assert!(eval_cache.stats.rows_sent <= eval_cache.stats.rows_requested);

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
            let active_seat = record["active_seat"].as_u64().unwrap() as usize;
            let exact_scores = record["exact_afterstate_score_active"].as_array().unwrap();
            let exact_components = record["exact_afterstate_score_decomposition_active"]
                .as_array()
                .unwrap();
            assert_eq!(exact_components.len(), action_count);
            for (score, components) in exact_scores.iter().zip(exact_components) {
                let component_sum: f64 = components
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|value| value.as_f64().unwrap())
                    .sum();
                assert!((component_sum - score.as_f64().unwrap()).abs() < 1e-6);
            }
            let final_score = record["final_score_vector"].clone();
            let final_component_sum: f64 = ["wildlife", "habitat", "nature_tokens"]
                .iter()
                .map(|category| {
                    record["score_decomposition"][active_seat.to_string()][category]
                        .as_f64()
                        .unwrap()
                })
                .sum();
            assert!(
                (final_component_sum - final_score[active_seat].as_f64().unwrap()).abs() < 1e-6
            );
            match &shared_final_score {
                None => shared_final_score = Some(final_score),
                Some(existing) => assert_eq!(
                    existing, &final_score,
                    "all records of a seed share the real outcome"
                ),
            }
        }

        let shard = ExpertTensorShardData::from_records(&records).expect("v4 shard");
        assert_eq!(shard.improved_policy_records, shard.record_count);
        assert_eq!(shard.improved_policy.len(), shard.total_action_count);
        assert_eq!(shard.search_root_value.len(), shard.record_count);
        assert_eq!(shard.exact_endgame_field_records, shard.record_count);
        assert_eq!(shard.exact_endgame.len(), shard.record_count);
        assert_eq!(shard.structured_value_field_records, shard.record_count);
        assert_eq!(
            shard.exact_afterstate_score_decomposition_active.len(),
            3 * shard.total_action_count
        );
        assert_eq!(shard.active_seat.len(), shard.record_count);

        // Full export path writes a v4 npz + manifest.
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
            Some(EXPERT_TENSOR_SCHEMA_ID_V4)
        );
        assert_eq!(
            manifest["version"].as_str(),
            Some(feature_tensors::EXPERT_SHARD_VERSION_V4)
        );
        assert!(manifest["created_unix_seconds"].as_u64().unwrap() > 0);
        assert_eq!(
            manifest["created_unix_seconds"],
            manifest["metadata"]["created_unix_seconds"]
        );
        assert!(manifest.get("created_at_utc").is_none());
        assert_eq!(manifest["metadata"]["ruleset_id"], RULESET_ID);
        assert_eq!(
            manifest["metadata"]["source_revision"],
            "test-source-revision"
        );
        assert_eq!(
            manifest["metadata"]["search"]["market_decision_samples"],
            args.gumbel_market_decision_samples
        );
        assert_eq!(
            manifest["metadata"]["search"]["exact_endgame_turns"],
            args.gumbel_exact_endgame_turns
        );
        assert_eq!(
            manifest["metadata"]["execution"]["seed_scheduler"],
            "dynamic_atomic_queue"
        );
        assert_eq!(
            manifest["metadata"]["teacher_model"]["checkpoint_tag"],
            "mock-checkpoint"
        );
        assert_eq!(manifest["metadata"]["teacher_model"]["step"], 17);
        assert_eq!(
            manifest["metadata"]["teacher_model"]["weights"]["sha256"],
            sha256_file_hex(&tempdir.join("mock.weights")).unwrap()
        );
        assert!(
            manifest["metadata"]["generator"]["sha256"]
                .as_str()
                .map_or(false, |hash| hash.len() == 64)
        );
        assert!(
            manifest["metadata"]["canonical_targets"]
                .as_array()
                .unwrap()
                .iter()
                .any(|target| target == "exact_endgame")
        );
        let archive = File::open(&args.out).expect("npz readable");
        let mut zip = zip::ZipArchive::new(archive).expect("npz zip");
        assert!(zip.by_name("exact_endgame.npy").is_ok());
        assert!(
            zip.by_name("exact_afterstate_score_decomposition_active.npy")
                .is_ok()
        );
        assert!(zip.by_name("active_seat.npy").is_ok());
        let _ = std::fs::remove_dir_all(&tempdir);
    }

    #[test]
    fn gumbel_selfplay_tensor_export_requires_source_revision() {
        let tempdir = std::env::temp_dir().join(format!(
            "cascadia-gumbel-source-revision-test-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&tempdir).expect("tempdir");
        let mut args = gumbel_test_args(&tempdir);
        args.source_revision = Some("   ".to_owned());
        let error = export_gumbel_selfplay_tensor_corpus(&args).unwrap_err();
        assert!(error.to_string().contains("source_revision"));
        let _ = std::fs::remove_dir_all(&tempdir);
    }

    #[test]
    fn packed_eval_request_roundtrips_feature_arrays() {
        use base64::Engine as _;

        let game = advanced_test_state(2_026_070_700, 6);
        let row = gumbel::eval_row_for_state(&game, Some(8))
            .expect("row")
            .expect("non-terminal");
        let raw = eval_request_for_row(&row, false).expect("raw request");
        let packed = eval_request_for_row(&row, true).expect("packed request");

        assert!(packed.get("legal_actions").is_none());
        assert!(packed.get("public_tokens").is_none());
        assert_eq!(packed["action_ids"], raw["action_ids"]);
        let features = packed.get("packed_features").expect("packed_features");
        let token_count = features["token_count"].as_u64().unwrap() as usize;
        let action_count = features["action_count"].as_u64().unwrap() as usize;
        assert_eq!(action_count, row.afterstates.len());

        let engine = base64::engine::general_purpose::STANDARD;
        let token_bytes = engine
            .decode(features["tokens_f32_b64"].as_str().unwrap())
            .expect("token b64");
        assert_eq!(
            token_bytes.len(),
            token_count * PUBLIC_TOKEN_FEATURE_DIM * 4
        );
        let decoded_tokens: Vec<f32> = token_bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        let expected_tokens: Vec<f32> = feature_tensors::public_token_features(&raw)
            .expect("expected tokens")
            .into_iter()
            .flatten()
            .collect();
        assert_eq!(decoded_tokens, expected_tokens, "token features bit-equal");

        let action_bytes = engine
            .decode(features["actions_f32_b64"].as_str().unwrap())
            .expect("action b64");
        assert_eq!(
            action_bytes.len(),
            action_count * SEMANTIC_PUBLIC_TOKEN_ACTION_FEATURE_DIM * 4
        );
        let tail_bytes = engine
            .decode(features["relation_tail_u8_b64"].as_str().unwrap())
            .expect("tail b64");
        assert_eq!(
            tail_bytes.len(),
            action_count * (token_count + action_count)
        );
        // The tail must contain at least one action-pointer relation on a
        // real mid-game state.
        assert!(tail_bytes.iter().any(|value| *value != 0));
    }

    #[test]
    fn action_relation_tail_fast_path_matches_reference() {
        for (seed, plies, menu) in [
            (2_026_070_700_u64, 2usize, 8usize),
            (2_026_070_700, 6, 8),
            (2_026_070_701, 10, 12),
            (2_026_070_702, 20, 6),
        ] {
            let state = advanced_test_state(seed, plies);
            let row = gumbel::eval_row_for_state(&state, Some(menu))
                .expect("row")
                .expect("non-terminal");
            let raw = eval_request_for_row(&row, false).expect("raw request");
            let token_count = feature_tensors::public_token_features(&raw)
                .expect("token features")
                .len();
            let action_count = row.afterstates.len();
            let fast = feature_tensors::action_relation_tail(&raw, token_count, action_count)
                .expect("fast tail");
            let reference =
                feature_tensors::action_relation_tail_reference(&raw, token_count, action_count)
                    .expect("reference tail");
            assert_eq!(
                fast, reference,
                "tail mismatch for seed {seed} plies {plies}"
            );
            assert!(fast.iter().any(|value| *value != 0));
        }
    }

    /// Deterministic bridge stand-in: eval values derived purely from the
    /// serialized request bytes, mirroring the real property that identical
    /// request payloads produce identical model outputs.
    fn synthetic_model_eval(request: &Value) -> ModelEval {
        let digest = eval_request_key(request, false).expect("request key");
        let action_count = request["action_ids"].as_array().expect("action ids").len();
        let priors: Vec<Value> = (0..action_count)
            .map(|index| json!((f64::from(digest[index % 32]) + 1.0) / 300.0))
            .collect();
        let score_to_go = (0..action_count)
            .map(|index| f64::from(digest[(index + 7) % 32]) / 8.0)
            .collect();
        ModelEval {
            priors,
            q: None,
            score_to_go: Some(score_to_go),
            value: None,
            model_fallback: false,
            response: json!({"type": "eval_response"}),
        }
    }

    /// Six rows over three distinct states: duplicates in mixed order.
    fn dedup_test_rows() -> Vec<gumbel::EvalRow> {
        let state_a = advanced_test_state(2_026_070_800, 2);
        let state_b = advanced_test_state(2_026_070_801, 3);
        let state_c = advanced_test_state(2_026_070_802, 4);
        let row = |state: &GameState| {
            gumbel::eval_row_for_state(state, Some(6))
                .expect("row")
                .expect("non-terminal")
        };
        vec![
            row(&state_a),
            row(&state_b),
            row(&state_a),
            row(&state_c),
            row(&state_b),
            row(&state_a),
        ]
    }

    #[test]
    fn deduped_eval_matches_undeduped_results_in_order() {
        let rows = dedup_test_rows();
        // Baseline: every row evaluated individually, no dedup or cache.
        let baseline: Vec<gumbel::EvalOut> = rows
            .iter()
            .map(|row| {
                let request = eval_request_for_row(row, false).expect("request");
                eval_out_for_row(row, &synthetic_model_eval(&request)).expect("eval out")
            })
            .collect();

        let mut cache = EvalRowCache::new();
        let mut bridge_rows = 0usize;
        let deduped = evaluate_rows_deduped(&rows, false, &mut cache, |requests, unique_rows| {
            bridge_rows += requests.len();
            assert_eq!(requests.len(), unique_rows.len());
            Ok(requests.iter().map(synthetic_model_eval).collect())
        })
        .expect("deduped eval");

        assert_eq!(deduped.len(), baseline.len());
        for (dedup_out, baseline_out) in deduped.iter().zip(baseline.iter()) {
            assert_eq!(dedup_out.priors, baseline_out.priors);
            assert_eq!(dedup_out.derived_final_q, baseline_out.derived_final_q);
        }
        assert_eq!(bridge_rows, 3, "six rows contain three unique states");
        assert_eq!(cache.stats.rows_requested, 6);
        assert_eq!(cache.stats.rows_sent, 3);
        assert_eq!(cache.stats.cache_hits, 0);
    }

    #[test]
    fn eval_cache_serves_repeat_calls_without_bridge_traffic() {
        let rows = dedup_test_rows();
        let mut cache = EvalRowCache::new();
        let first = evaluate_rows_deduped(&rows, false, &mut cache, |requests, _| {
            Ok(requests.iter().map(synthetic_model_eval).collect())
        })
        .expect("first eval");
        let second = evaluate_rows_deduped(&rows, false, &mut cache, |_, _| {
            panic!("cache-served batch must not reach the bridge")
        })
        .expect("second eval");

        assert_eq!(first.len(), second.len());
        for (fresh, cached) in first.iter().zip(second.iter()) {
            assert_eq!(fresh.priors, cached.priors);
            assert_eq!(fresh.derived_final_q, cached.derived_final_q);
        }
        assert_eq!(cache.stats.rows_requested, 2 * rows.len() as u64);
        assert_eq!(cache.stats.rows_sent, 3);
        assert_eq!(cache.stats.cache_hits, rows.len() as u64);
    }

    fn clone_eval_row(row: &gumbel::EvalRow) -> gumbel::EvalRow {
        gumbel::EvalRow {
            staged: row.staged.clone(),
            prelude: row.prelude.clone(),
            afterstates: row.afterstates.clone(),
        }
    }

    /// Corpus of rows with deliberate near-duplicates: identical rows,
    /// redeterminized hidden state, shared states with different menus, and
    /// single-field perturbations of every request-visible input.
    fn key_completeness_corpus() -> Vec<gumbel::EvalRow> {
        let mut rows = Vec::new();
        for (seed, plies) in [
            (2_026_070_900_u64, 2usize),
            (2_026_070_901, 4),
            (2_026_070_902, 7),
        ] {
            let state = advanced_test_state(seed, plies);
            for menu in [4usize, 6] {
                rows.push(
                    gumbel::eval_row_for_state(&state, Some(menu))
                        .expect("row")
                        .expect("non-terminal"),
                );
            }
        }

        let base = clone_eval_row(&rows[0]);

        // Byte-identical duplicate.
        rows.push(clone_eval_row(&base));

        // Same public state, permuted hidden order: requests stay identical.
        let mut redet = clone_eval_row(&base);
        redet
            .staged
            .redeterminize_hidden(GameSeed::from_u64(0xfeed_f00d));
        rows.push(redet);

        // Exact afterstate score perturbation.
        let mut scored = clone_eval_row(&base);
        scored.afterstates[0].exact_score_active += 0.5;
        rows.push(scored);

        // Immediate base-score perturbation.
        let mut base_scored = clone_eval_row(&base);
        base_scored.afterstates[0].candidate.resulting_base_score += 1;
        rows.push(base_scored);

        // Prelude projection perturbation (feeds every cleanup_choice field).
        let mut prelude_flip = clone_eval_row(&base);
        prelude_flip.prelude.replace_three_of_a_kind =
            !prelude_flip.prelude.replace_three_of_a_kind;
        rows.push(prelude_flip);

        // Action identity perturbation (only the serialized action changes).
        let mut action_flip = clone_eval_row(&base);
        action_flip.afterstates[0]
            .candidate
            .action
            .replace_three_of_a_kind = !action_flip.afterstates[0]
            .candidate
            .action
            .replace_three_of_a_kind;
        rows.push(action_flip);

        // Menu order perturbation.
        let mut swapped = clone_eval_row(&base);
        swapped.afterstates.swap(0, 1);
        rows.push(swapped);

        // Menu subset perturbation.
        let mut truncated = clone_eval_row(&base);
        truncated.afterstates.pop();
        rows.push(truncated);

        // All non-finite exact scores collapse to JSON null: NaN and +inf
        // rows must produce identical requests AND identical keys.
        let mut nan_row = clone_eval_row(&base);
        nan_row.afterstates[0].exact_score_active = f64::NAN;
        rows.push(nan_row);
        let mut inf_row = clone_eval_row(&base);
        inf_row.afterstates[0].exact_score_active = f64::INFINITY;
        rows.push(inf_row);

        rows
    }

    #[test]
    fn eval_row_key_matches_full_request_equality() {
        let rows = key_completeness_corpus();
        let fingerprints: Vec<([u8; 32], [u8; 32], Vec<u8>)> = rows
            .iter()
            .map(|row| {
                let raw_key = eval_row_key(row, false).expect("raw key").0;
                let packed_key = eval_row_key(row, true).expect("packed key").0;
                let request = eval_request_for_row(row, false).expect("request");
                let bytes = serde_json::to_vec(&request).expect("request bytes");
                assert_ne!(raw_key, packed_key, "wire marker must split key spaces");
                (raw_key, packed_key, bytes)
            })
            .collect();

        let mut equal_pairs = 0usize;
        for left in 0..fingerprints.len() {
            for right in left + 1..fingerprints.len() {
                let requests_equal = fingerprints[left].2 == fingerprints[right].2;
                assert_eq!(
                    fingerprints[left].0 == fingerprints[right].0,
                    requests_equal,
                    "raw key equality must match request byte equality for rows {left} and {right}"
                );
                assert_eq!(
                    fingerprints[left].1 == fingerprints[right].1,
                    requests_equal,
                    "packed key equality must match request byte equality for rows {left} and {right}"
                );
                if requests_equal {
                    equal_pairs += 1;
                }
            }
        }
        // The corpus must exercise the equal side too: the duplicate, the
        // redeterminized twin, and the NaN/inf pair.
        assert!(
            equal_pairs >= 4,
            "expected at least 4 request-equal pairs, found {equal_pairs}"
        );
    }

    #[test]
    fn rotated_eval_rows_preserve_menus_and_exact_scores() {
        let state = advanced_test_state(2_026_070_901, 6);
        let row = gumbel::eval_row_for_state(&state, Some(8))
            .expect("row")
            .expect("non-terminal");
        for steps in 1..6u8 {
            let rotated = rotated_eval_row(&row, steps);
            assert_eq!(rotated.afterstates.len(), row.afterstates.len());
            for (original, transformed) in row.afterstates.iter().zip(rotated.afterstates.iter()) {
                assert_eq!(
                    original.exact_score_active, transformed.exact_score_active,
                    "exact scoring is rotation-invariant"
                );
                assert_eq!(original.apply_truncated, transformed.apply_truncated);
                assert_eq!(
                    original.candidate.resulting_base_score,
                    transformed.candidate.resulting_base_score
                );
            }
            // Both frames must build valid eval requests with equal menu sizes.
            let original_request = eval_request_for_row(&row, false).expect("request");
            let rotated_request = eval_request_for_row(&rotated, false).expect("rotated request");
            assert_eq!(
                original_request["action_ids"].as_array().unwrap().len(),
                rotated_request["action_ids"].as_array().unwrap().len()
            );
            // Different frame -> different public state hash (the whole point).
            assert_ne!(
                original_request["state_hash"],
                rotated_request["state_hash"]
            );
        }
    }

    #[test]
    fn average_model_evals_takes_elementwise_means() {
        let make = |prior: f64, stg: f64| ModelEval {
            priors: vec![json!(prior), json!(1.0 - prior)],
            q: Some(vec![stg + 1.0, stg + 2.0]),
            score_to_go: Some(vec![stg, stg * 2.0]),
            value: Some(vec![stg; 4]),
            model_fallback: false,
            response: json!({}),
        };
        let a = make(0.25, 1.0);
        let b = make(0.75, 3.0);
        let averaged = average_model_evals(&[&a, &b]).expect("average");
        assert_eq!(averaged.priors[0].as_f64().unwrap(), 0.5);
        assert_eq!(averaged.priors[1].as_f64().unwrap(), 0.5);
        assert_eq!(averaged.score_to_go.as_ref().unwrap()[0], 2.0);
        assert_eq!(averaged.score_to_go.as_ref().unwrap()[1], 4.0);
        assert_eq!(averaged.q.as_ref().unwrap()[0], 3.0);
        assert_eq!(averaged.value.as_ref().unwrap()[2], 2.0);
        // Missing optional in any variant -> None.
        let mut c = make(0.5, 2.0);
        c.value = None;
        let partial = average_model_evals(&[&a, &c]).expect("average");
        assert!(partial.value.is_none());
        assert!(partial.score_to_go.is_some());
    }

    #[test]
    fn gumbel_policy_game_runs_with_tta_rotations() {
        let tempdir =
            std::env::temp_dir().join(format!("cascadia-gumbel-tta-test-{}", std::process::id()));
        std::fs::create_dir_all(&tempdir).expect("tempdir");
        let mut args = gumbel_test_args(&tempdir);
        args.mode = Mode::GumbelPolicyGame;
        args.out = tempdir.join("gumbel_tta_game.jsonl");
        args.seed_count = 1;
        args.gumbel_exploration = false;
        args.gumbel_tta = 3;

        run_gumbel_policy_game(&args).expect("tta policy game runs");
        let contents = std::fs::read_to_string(&args.out).expect("game jsonl");
        let done = contents
            .lines()
            .map(|line| serde_json::from_str::<Value>(line).expect("jsonl line"))
            .find(|line| line["type"] == "gumbel_game_done")
            .expect("done record");
        assert_eq!(done["scores"].as_array().unwrap().len(), 4);
        let _ = std::fs::remove_dir_all(&tempdir);
    }

    #[test]
    fn gumbel_policy_game_emits_decisions_and_done() {
        let tempdir =
            std::env::temp_dir().join(format!("cascadia-gumbel-game-test-{}", std::process::id()));
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

    /// Plays one tiny mock-bridge policy game and returns (args, ledger
    /// lines). Shared fixture for the ledger-replay modes' tests.
    fn play_tiny_policy_game(tempdir: &std::path::Path, seed: u64) -> (Args, Vec<Value>) {
        let mut args = gumbel_test_args(tempdir);
        args.mode = Mode::GumbelPolicyGame;
        args.out = tempdir.join(format!("ledger_game_{seed}.jsonl"));
        args.first_seed = seed;
        args.seed_count = 1;
        args.gumbel_exploration = false;
        run_gumbel_policy_game(&args).expect("policy game runs");
        let contents = std::fs::read_to_string(&args.out).expect("game jsonl");
        let lines: Vec<Value> = contents
            .lines()
            .map(|line| serde_json::from_str(line).expect("jsonl line"))
            .collect();
        (args, lines)
    }

    #[test]
    fn ledger_replay_reconstructs_policy_game_trajectory() {
        let tempdir = std::env::temp_dir().join(format!(
            "cascadia-ledger-replay-test-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&tempdir).expect("tempdir");
        let seed = 2_026_070_777_u64;
        let (args, lines) = play_tiny_policy_game(&tempdir, seed);
        let decision_count = lines
            .iter()
            .filter(|line| line["type"] == "gumbel_decision")
            .count();
        let done = lines
            .iter()
            .find(|line| line["type"] == "gumbel_game_done")
            .expect("done record");

        let by_seed = read_ledger_decision_rows(&args.out).expect("ledger parses");
        assert_eq!(by_seed.len(), 1);
        let rows = by_seed.get(&seed).expect("seed present");
        assert_eq!(rows.len(), decision_count);
        let (eval_rows, metas, final_state) =
            replay_ledger_seed(seed, rows, gumbel_root_menu_limit(&args), args.player_count)
                .expect("replay succeeds");
        assert_eq!(eval_rows.len(), decision_count);
        assert_eq!(metas.len(), decision_count);
        assert!(metas.iter().all(|meta| !meta.full_menu_fallback));

        // The replayed trajectory must land on exactly the game the ledger
        // scored: per-seat exact totals equal the done record's totals.
        let done_totals: Vec<f64> = done["scores"]
            .as_array()
            .expect("scores array")
            .iter()
            .map(|score| score["total"].as_f64().expect("total"))
            .collect();
        let replay_totals: Vec<f64> = score_game(&final_state)
            .iter()
            .map(|score| f64::from(score.total))
            .collect();
        assert_eq!(done_totals, replay_totals);
        let _ = std::fs::remove_dir_all(&tempdir);
    }

    #[test]
    fn table_contention_audit_runs_on_mock_bridge() {
        let tempdir = std::env::temp_dir().join(format!(
            "cascadia-contention-audit-test-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&tempdir).expect("tempdir");
        let seed = 2_026_070_778_u64;
        let (mut args, _lines) = play_tiny_policy_game(&tempdir, seed);
        args.mode = Mode::TableContentionAudit;
        args.input = Some(args.out.clone());
        args.out = tempdir.join("contention_audit.jsonl");
        run_table_contention_audit(&args).expect("audit runs");

        let contents = std::fs::read_to_string(&args.out).expect("audit jsonl");
        let lines: Vec<Value> = contents
            .lines()
            .map(|line| serde_json::from_str(line).expect("jsonl line"))
            .collect();
        let summary = lines
            .iter()
            .find(|line| line["type"] == "contention_summary")
            .expect("summary record");
        let audited = summary["decisions_audited"].as_u64().expect("count");
        assert!(audited > 0);
        for line in lines.iter().filter(|line| line["type"] == "contention_decision") {
            assert_ne!(line["chosen"]["index"], line["runner"]["index"]);
            assert!(line["chosen_table"].as_f64().expect("chosen table").is_finite());
            assert!(line["runner_table"].as_f64().expect("runner table").is_finite());
            let delta = line["table_delta_runner_minus_chosen"]
                .as_f64()
                .expect("delta");
            let reconstructed = line["runner_table"].as_f64().unwrap()
                - line["chosen_table"].as_f64().unwrap();
            assert!((delta - reconstructed).abs() < 1e-9);
        }
        let _ = std::fs::remove_dir_all(&tempdir);
    }

    #[test]
    fn search_stability_probe_pairs_search_seeds_across_variants() {
        let tempdir = std::env::temp_dir().join(format!(
            "cascadia-stability-probe-test-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&tempdir).expect("tempdir");
        let seed = 2_026_070_779_u64;
        let (mut args, _lines) = play_tiny_policy_game(&tempdir, seed);
        args.mode = Mode::SearchStabilityProbe;
        args.input = Some(args.out.clone());
        args.out = tempdir.join("stability_probe.jsonl");
        args.probe_stride = 1;
        args.probe_repeats = 2;
        args.probe_max_roots = 2;
        args.gumbel_blend_weight = 0.5;
        run_search_stability_probe(&args).expect("probe runs");

        let contents = std::fs::read_to_string(&args.out).expect("probe jsonl");
        let lines: Vec<Value> = contents
            .lines()
            .map(|line| serde_json::from_str(line).expect("jsonl line"))
            .collect();
        let searches: Vec<&Value> = lines
            .iter()
            .filter(|line| line["type"] == "stability_search")
            .collect();
        // 2 roots x 2 variants x 2 repeats.
        assert_eq!(searches.len(), 8);
        let summary = lines
            .iter()
            .find(|line| line["type"] == "stability_summary")
            .expect("summary record");
        assert_eq!(summary["sampled_roots"].as_u64(), Some(2));

        // Equal (ply, repeat) must share one search seed across the two
        // variants (meta-level CRN), and each root must contribute both
        // variants.
        let mut seeds_by_key: HashMap<(u64, u64), Vec<(bool, u64)>> = HashMap::new();
        for row in &searches {
            let key = (
                row["ply"].as_u64().expect("ply"),
                row["repeat"].as_u64().expect("repeat"),
            );
            seeds_by_key.entry(key).or_default().push((
                row["paired_rollouts"].as_bool().expect("variant"),
                row["search_seed"].as_u64().expect("search seed"),
            ));
        }
        for ((_ply, _repeat), entries) in seeds_by_key {
            assert_eq!(entries.len(), 2);
            assert_ne!(entries[0].0, entries[1].0, "one row per variant");
            assert_eq!(entries[0].1, entries[1].1, "shared search seed");
        }
        let _ = std::fs::remove_dir_all(&tempdir);
    }

    #[test]
    fn puzzle_bank_resolves_roots_into_per_seed_shards() {
        let tempdir = std::env::temp_dir().join(format!(
            "cascadia-puzzle-bank-test-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&tempdir).expect("tempdir");
        let seed = 2_026_070_781_u64;
        let (mut args, _lines) = play_tiny_policy_game(&tempdir, seed);
        args.mode = Mode::PuzzleBank;
        args.input = Some(args.out.clone());
        args.output_dir = Some(tempdir.join("bank"));
        args.probe_stride = 3;
        args.probe_repeats = 2;
        args.model_sessions = Some(2);
        run_puzzle_bank(&args).expect("puzzle bank runs");

        let shard = tempdir.join("bank").join(format!("puzzle_seed_{seed}.jsonl"));
        let contents = std::fs::read_to_string(&shard).expect("shard exists");
        let mut roots = 0usize;
        for line in contents.lines() {
            let record: Value = serde_json::from_str(line).expect("jsonl line");
            assert_eq!(record["type"], "puzzle_root");
            let action_count = record["action_count"].as_u64().expect("count") as usize;
            assert_eq!(record["action_ids"].as_array().unwrap().len(), action_count);
            assert_eq!(
                record["mean_completed_q"].as_array().unwrap().len(),
                action_count
            );
            assert_eq!(record["repeats"].as_u64(), Some(2));
            assert_eq!(
                record["repeat_chosen_indexes"].as_array().unwrap().len(),
                2
            );
            roots += 1;
        }
        assert!(roots > 0, "at least one root resolved");
        let manifest_path = tempdir.join("bank").join("puzzle_bank_manifest.json");
        let manifest: Value = serde_json::from_str(
            &std::fs::read_to_string(&manifest_path).expect("manifest"),
        )
        .expect("manifest json");
        assert_eq!(manifest["resolved_roots"].as_u64(), Some(roots as u64));
        assert_eq!(manifest["repeats"].as_u64(), Some(2));
        let _ = std::fs::remove_dir_all(&tempdir);
    }

    /// Strips wall-clock timing fields so record content can be compared
    /// across runs. Everything else in the decision/done records is a pure
    /// function of args + seed + bridge outputs.
    fn normalize_game_records(lines: &[Value]) -> Vec<Value> {
        lines
            .iter()
            .cloned()
            .map(|mut line| {
                if let Some(object) = line.as_object_mut() {
                    object.remove("decision_seconds");
                    object.remove("elapsed_seconds");
                }
                line
            })
            .collect()
    }

    fn read_jsonl_values(path: &std::path::Path) -> Vec<Value> {
        std::fs::read_to_string(path)
            .unwrap_or_else(|error| panic!("reading {}: {error}", path.display()))
            .lines()
            .map(|line| serde_json::from_str(line).expect("jsonl line"))
            .collect()
    }

    #[test]
    fn dynamic_seed_workers_backfill_while_a_long_seed_is_running() {
        use std::sync::Condvar;
        use std::time::Duration;

        let long_seed_release = Arc::new((Mutex::new(false), Condvar::new()));
        let release_for_run = Arc::clone(&long_seed_release);
        let (mut outputs, workers) = run_dynamic_seed_workers(
            &[0, 1, 2],
            2,
            |worker_index| Ok(worker_index),
            move |worker_index, seed| {
                if seed == 0 {
                    let (lock, wake) = &*release_for_run;
                    let mut released = lock.lock().expect("release lock");
                    while !*released {
                        let (next, timeout) = wake
                            .wait_timeout(released, Duration::from_secs(5))
                            .expect("release wait");
                        released = next;
                        if timeout.timed_out() && !*released {
                            bail!("dynamic worker failed to backfill seed 2");
                        }
                    }
                } else if seed == 2 {
                    let (lock, wake) = &*release_for_run;
                    *lock.lock().expect("release lock") = true;
                    wake.notify_all();
                }
                Ok(*worker_index)
            },
        )
        .expect("dynamic schedule");
        outputs.sort_by_key(|(seed, _)| *seed);

        assert_eq!(workers.len(), 2);
        assert_eq!(outputs.len(), 3);
        assert_ne!(outputs[0].1, outputs[1].1, "first two seeds start apart");
        assert_eq!(
            outputs[1].1, outputs[2].1,
            "worker finishing short seed 1 must backfill seed 2 while seed 0 runs"
        );
    }

    #[test]
    fn gumbel_benchmark_batch_matches_single_seed_policy_games() {
        let tempdir =
            std::env::temp_dir().join(format!("cascadia-gumbel-batch-test-{}", std::process::id()));
        std::fs::create_dir_all(&tempdir).expect("tempdir");
        let first_seed = 2_026_070_600u64;

        // Reference: one --gumbel-policy-game run per seed, each owning its
        // bridge session, exactly like the current Python harness default.
        let mut single_lines_by_seed: HashMap<u64, Vec<Value>> = HashMap::new();
        for seed_u64 in first_seed..first_seed + 2 {
            let mut single_args = gumbel_test_args(&tempdir);
            single_args.mode = Mode::GumbelPolicyGame;
            single_args.gumbel_exploration = false;
            single_args.first_seed = seed_u64;
            single_args.seed_count = 1;
            single_args.out = tempdir.join(format!("single_seed_{seed_u64}.jsonl"));
            run_gumbel_policy_game(&single_args).expect("single-seed policy game");
            single_lines_by_seed.insert(seed_u64, read_jsonl_values(&single_args.out));
        }

        // Candidate: both seeds through one batch run over the shared
        // aggregated bridge (two parallel game workers, one bridge process).
        let mut batch_args = gumbel_test_args(&tempdir);
        batch_args.mode = Mode::GumbelBenchmarkBatch;
        batch_args.gumbel_exploration = false;
        batch_args.first_seed = first_seed;
        batch_args.seed_count = 2;
        batch_args.model_sessions = Some(2);
        batch_args.shared_model_session = true;
        let batch_dir = tempdir.join("batch_out");
        batch_args.output_dir = Some(batch_dir.clone());
        run_gumbel_benchmark_batch(&batch_args).expect("benchmark batch runs");

        for seed_u64 in first_seed..first_seed + 2 {
            let batch_path = batch_dir.join(format!("gumbel_game_seed_{seed_u64}.jsonl"));
            assert!(
                batch_path.exists(),
                "batch mode must write per-seed JSONL for seed {seed_u64}"
            );
            let batch_lines = read_jsonl_values(&batch_path);
            let single_lines = &single_lines_by_seed[&seed_u64];
            assert_eq!(
                normalize_game_records(&batch_lines),
                normalize_game_records(single_lines),
                "seed {seed_u64}: batch records must equal single-seed records \
                 field-for-field (timing fields excluded)"
            );
            // The done record must really be a played game, not a stub.
            let done = batch_lines
                .iter()
                .find(|line| line["type"] == "gumbel_game_done")
                .expect("done record");
            assert!(done["decision_count"].as_u64().unwrap() > 0);
        }
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
