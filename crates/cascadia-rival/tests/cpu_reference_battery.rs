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

#[test]
fn pr_sized_dense_cpu_reference_battery_has_zero_mismatches() {
    let started = Instant::now();
    let config = GameConfig::research_aaaaa(4).expect("four-player AAAAA is canonical");
    let compiler = DenseSemanticCompiler;
    let mut completed_games = 0_u64;
    let mut transition_checks = 0_u64;

    for game_index in 0..COMPLETE_GAMES {
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
        assert_terminal_ledger_roundtrip(&sealed, &game, &terminal_scores, game_index == 0);
        completed_games += 1;
    }

    assert_eq!(completed_games, COMPLETE_GAMES);
    assert_eq!(transition_checks, EXPECTED_TRANSITION_CHECKS);
    eprintln!(
        "Rival PR CPU reference battery: {completed_games} complete games, \
         {transition_checks} transition/compiler parity checks, zero mismatches in {:.3}s",
        started.elapsed().as_secs_f64()
    );
}
