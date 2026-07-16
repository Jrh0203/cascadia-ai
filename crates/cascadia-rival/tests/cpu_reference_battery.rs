//! PR-sized deterministic battery for the Rival dense CPU reference path.
//!
//! This is deliberately a test, not an experiment: fixed local seeds drive
//! random legal play, and every assertion compares Rival's dense semantic
//! compiler and replay ledger with the canonical `cascadia-game` engine.

use std::time::Instant;

use cascadia_game::{GameConfig, GameSeed, GameState, ScoreBreakdown, score_game};
use cascadia_rival::{
    DENSE_COMPILER_ID, DenseSemanticCompiler, SemanticCompiler, TrajectoryLedger,
    TrajectoryLedgerBuilder,
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

const COMPLETE_GAMES: u64 = 125;
const TRANSITIONS_PER_GAME: u16 = 80;
const EXPECTED_TRANSITION_CHECKS: u64 = 10_000;
const GAME_SEED_BASE: u64 = 0x5249_5641_4c00_0000;
const ACTION_SEED_BASE: u64 = 0x5249_5641_4c80_0000;
/// CPU-1 release scale: 12,500 four-player games x 80 transitions each =
/// exactly 1,000,000 transition/compiler parity checks. The plan's original
/// "10,000 games and 1,000,000 transitions" was internally inconsistent at
/// 80 transitions per game; the stronger number wins (build scope WI-1).
const RELEASE_GAMES_DEFAULT: u64 = 12_500;
const RELEASE_GAMES_ENV: &str = "RIVAL_BATTERY_GAMES";
const RELEASE_REPORT_ENV: &str = "RIVAL_BATTERY_REPORT";
/// Full JSON-boundary verification (serialize + decode + replay) is run on
/// every JSON_BOUNDARY_STRIDE-th game; every game still gets the sealed
/// replay + hash verification inside `seal_terminal`.
const JSON_BOUNDARY_STRIDE: u64 = 500;

fn assert_score_decomposition(score: ScoreBreakdown) {
    let base_total = score.habitat.into_iter().sum::<u16>()
        + score.wildlife.into_iter().sum::<u16>()
        + score.nature_tokens;
    let habitat_bonus = score.habitat_bonus.into_iter().map(u16::from).sum::<u16>();
    assert_eq!(score.base_total, base_total);
    assert_eq!(score.total, base_total + habitat_bonus);
    assert_eq!(score.habitat_bonus, [0; 5]);
}

fn assert_terminal_ledger_roundtrip(
    ledger: &TrajectoryLedger,
    expected: &GameState,
    expected_scores: &[ScoreBreakdown],
    check_json_boundary: bool,
) {
    assert_eq!(ledger.turns().len(), usize::from(TRANSITIONS_PER_GAME));
    assert_eq!(
        ledger.final_state_hash(),
        expected.canonical_hash().as_bytes()
    );
    assert_eq!(ledger.terminal_scores(), Some(expected_scores));

    // `seal_terminal` has already replayed and hash-verified every turn before
    // returning this value. One complete trajectory additionally crosses the
    // substantially more expensive JSON verifier and an explicit replay;
    // byte-level serialization has its own focused exhaustive unit tests.
    if check_json_boundary {
        let replayed = ledger.replay().expect("sealed terminal ledger must replay");
        assert_eq!(replayed, *expected);
        assert!(replayed.is_game_over());
        let encoded = ledger
            .canonical_json_bytes()
            .expect("verified terminal ledger must serialize");
        let decoded = TrajectoryLedger::from_json_slice(&encoded)
            .expect("canonical terminal ledger JSON must verify on read");
        assert_eq!(&decoded, ledger);
        assert_eq!(
            decoded.replay().expect("decoded ledger must replay"),
            *expected
        );
    }
}

/// Runs `games` complete random-legal-play games with full canonical/dense
/// parity checks, returning (completed_games, transition_checks, elapsed_s).
fn run_dense_battery(games: u64, json_boundary_stride: u64) -> (u64, u64, f64) {
    let started = Instant::now();
    let config = GameConfig::research_aaaaa(4).expect("four-player AAAAA is canonical");
    let compiler = DenseSemanticCompiler;
    let mut completed_games = 0_u64;
    let mut transition_checks = 0_u64;

    for game_index in 0..games {
        let seed = GameSeed::from_u64(GAME_SEED_BASE + game_index);
        let mut game = GameState::new(config, seed).expect("fixed seed must initialize");
        let mut ledger = TrajectoryLedgerBuilder::new(
            format!("rival-pr-cpu-battery-{game_index:03}"),
            game.clone(),
        )
        .expect("canonical source must initialize a ledger");
        let mut action_rng = ChaCha8Rng::seed_from_u64(ACTION_SEED_BASE + game_index);

        assert_eq!(game.total_turns(), TRANSITIONS_PER_GAME);
        while !game.is_game_over() {
            game.validate()
                .expect("reachable source state must be valid");
            assert_eq!(ledger.game(), &game);

            let actor = game.current_player();
            let source_hash = *game.canonical_hash().as_bytes();
            let source_scores = score_game(&game);
            source_scores
                .iter()
                .copied()
                .for_each(assert_score_decomposition);

            let state_semantics = compiler
                .compile_state(&game)
                .expect("dense state compilation must accept a reachable state");
            assert_eq!(state_semantics.compiler_id, DENSE_COMPILER_ID);
            assert_eq!(state_semantics.state_hash, source_hash);
            assert_eq!(state_semantics.current_player, actor);
            assert_eq!(state_semantics.completed_turns, game.completed_turns());
            assert_eq!(state_semantics.terminal, game.is_game_over());
            assert_eq!(state_semantics.scores, source_scores);

            let prelude_choices = game
                .free_three_of_a_kind_choices()
                .expect("reachable state must expose legal prelude choices");
            assert!(!prelude_choices.is_empty());
            let prelude = &prelude_choices[action_rng.gen_range(0..prelude_choices.len())];
            let actions = game
                .legal_turn_actions(prelude)
                .expect("chosen canonical prelude must enumerate legal turns");
            assert!(!actions.is_empty());
            let action = actions[action_rng.gen_range(0..actions.len())].clone();
            assert_eq!(&action.prelude(), prelude);

            let action_semantics = compiler
                .compile_action(&game, &action)
                .expect("dense action compilation must accept a legal turn");
            let transitioned = game
                .transition(&action)
                .expect("materialized legal action must transition");
            let mut applied = game.clone();
            applied
                .apply(&action)
                .expect("clone-plus-apply must accept the same legal turn");
            assert_eq!(applied, transitioned);
            assert_eq!(applied.canonical_bytes(), transitioned.canonical_bytes());
            assert_eq!(applied.canonical_hash(), transitioned.canonical_hash());

            let after_scores = score_game(&transitioned);
            after_scores
                .iter()
                .copied()
                .for_each(assert_score_decomposition);
            assert_eq!(action_semantics.compiler_id, DENSE_COMPILER_ID);
            assert_eq!(action_semantics.actor, actor);
            assert_eq!(action_semantics.source_state_hash, source_hash);
            assert_eq!(
                action_semantics.after_state_hash,
                *transitioned.canonical_hash().as_bytes()
            );
            assert_eq!(action_semantics.before, source_scores[actor]);
            assert_eq!(action_semantics.after, after_scores[actor]);
            assert_eq!(
                action_semantics.own_score_delta.category_sum(),
                action_semantics.own_score_delta.total
            );

            ledger
                .push_fixture_turn(action)
                .expect("ledger must accept the canonical legal turn");
            assert_eq!(ledger.game(), &transitioned);
            game = transitioned;
            transition_checks += 1;
        }

        assert_eq!(game.completed_turns(), TRANSITIONS_PER_GAME);
        assert!(game.is_game_over());
        game.validate()
            .expect("terminal reachable state must be valid");
        let terminal_scores = score_game(&game);
        terminal_scores
            .iter()
            .copied()
            .for_each(assert_score_decomposition);
        let terminal_semantics = compiler
            .compile_state(&game)
            .expect("dense state compilation must accept a terminal state");
        assert!(terminal_semantics.terminal);
        assert_eq!(terminal_semantics.scores, terminal_scores);

        let sealed = ledger
            .seal_terminal()
            .expect("complete canonical trajectory must seal as terminal");
        assert_terminal_ledger_roundtrip(
            &sealed,
            &game,
            &terminal_scores,
            game_index % json_boundary_stride == 0,
        );
        completed_games += 1;
    }

    (completed_games, transition_checks, started.elapsed().as_secs_f64())
}

#[test]
fn pr_sized_dense_cpu_reference_battery_has_zero_mismatches() {
    let (completed_games, transition_checks, elapsed) =
        run_dense_battery(COMPLETE_GAMES, COMPLETE_GAMES);
    assert_eq!(completed_games, COMPLETE_GAMES);
    assert_eq!(transition_checks, EXPECTED_TRANSITION_CHECKS);
    eprintln!(
        "Rival PR CPU reference battery: {completed_games} complete games, \
         {transition_checks} transition/compiler parity checks, zero mismatches in {elapsed:.3}s",
    );
}

/// CPU-1 release battery (build scope WI-1). Deliberately `#[ignore]`d: run
/// explicitly with
/// `cargo test -p cascadia-rival --release --test cpu_reference_battery -- --ignored`.
/// Every assertion is identical to the PR battery; only the scale and the
/// durable JSON receipt differ. A panic anywhere means CPU-1 is NOT claimed.
#[test]
#[ignore = "release-scale CPU-1 battery: hours of CPU; invoke explicitly"]
fn release_scale_dense_cpu_reference_battery_has_zero_mismatches() {
    let games = std::env::var(RELEASE_GAMES_ENV)
        .ok()
        .map(|value| {
            value
                .parse::<u64>()
                .expect("RIVAL_BATTERY_GAMES must be a positive integer")
        })
        .unwrap_or(RELEASE_GAMES_DEFAULT);
    assert!(games > 0, "release battery requires at least one game");
    let (completed_games, transition_checks, elapsed) =
        run_dense_battery(games, JSON_BOUNDARY_STRIDE);
    assert_eq!(completed_games, games);
    assert_eq!(transition_checks, games * u64::from(TRANSITIONS_PER_GAME));

    let report = serde_json::json!({
        "schema_id": "cascadiav3.rival_cpu1_battery_receipt.v1",
        "battery": "release_scale_dense_cpu_reference_battery",
        "games": completed_games,
        "transitions_per_game": TRANSITIONS_PER_GAME,
        "transition_checks": transition_checks,
        "json_boundary_stride": JSON_BOUNDARY_STRIDE,
        "game_seed_base": GAME_SEED_BASE,
        "action_seed_base": ACTION_SEED_BASE,
        "elapsed_seconds": elapsed,
        "zero_mismatches": true,
        "notes": "Every transition compared canonical engine vs dense semantic compiler; every game sealed, replayed, and hash-verified; JSON boundary round-trip on every stride-th game.",
    });
    let report_path = std::env::var(RELEASE_REPORT_ENV)
        .unwrap_or_else(|_| "rival_cpu1_battery_receipt.json".to_owned());
    std::fs::write(
        &report_path,
        format!("{}\n", serde_json::to_string_pretty(&report).expect("report serializes")),
    )
    .expect("release battery receipt must be written");
    eprintln!(
        "Rival CPU-1 release battery: {completed_games} games, {transition_checks} checks, \
         zero mismatches in {elapsed:.1}s -> {report_path}",
    );
}
